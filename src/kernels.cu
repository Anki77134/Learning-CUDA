#include <vector>
#include <cuda_fp16.h>
#include <cmath>
#include <algorithm>
#include <limits>

#include "../tester/utils.h"

// 用于并行归约计算迹的内核
// 每个块计算对角线元素的一个部分
template <typename T>
__global__ void traceKernel(const T* input, T* output, size_t rows, size_t cols) {
  extern __shared__ char shared_mem[];
  T* sdata = reinterpret_cast<T*>(shared_mem);

  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t diag_size = min(rows, cols);

  // 将对角元素加载到共享内存中（如果超出范围，则为0）
  if (idx < diag_size) {
    sdata[tid] = input[idx * cols + idx];
  } else {
    sdata[tid] = T(0);
  }
  __syncthreads();

  // 在共享内存中执行树形归约
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // 线程0将结果写入该块
  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // 计算对角线的大小
  size_t diag_size = std::min(rows, cols);

  if (diag_size == 0) {
    return T(0);
  }

  // 分配设备内存
  T* d_input;
  T* d_partial_sums;

  size_t input_size = rows * cols;
  int threads_per_block = 256;
  int num_blocks = (diag_size + threads_per_block - 1) / threads_per_block;

  RUNTIME_CHECK(cudaMalloc(&d_input, input_size * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_partial_sums, num_blocks * sizeof(T)));

  // 将输入数据拷贝到设备
  RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(T), cudaMemcpyHostToDevice));

  // 启动内核计算部分和
  size_t shared_mem_size = threads_per_block * sizeof(T);
  traceKernel<<<num_blocks, threads_per_block, shared_mem_size>>>(d_input, d_partial_sums, rows, cols);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());

  // 将部分和从设备拷贝回主机并计算最终的和
  std::vector<T> h_partial_sums(num_blocks);
  RUNTIME_CHECK(cudaMemcpy(h_partial_sums.data(), d_partial_sums, num_blocks * sizeof(T), cudaMemcpyDeviceToHost));

  T result = T(0);
  for (int i = 0; i < num_blocks; i++) {
    result += h_partial_sums[i];
  }

  // 释放设备内存
  RUNTIME_CHECK(cudaFree(d_input));
  RUNTIME_CHECK(cudaFree(d_partial_sums));

  return result;
}

// 类型转换的辅助函数
__device__ __forceinline__ float toFloat(float x) { return x; }
__device__ __forceinline__ float toFloat(half x) { return __half2float(x); }

/**
 * Flash Attention 内核 - 每个块处理一个查询位置的一个头
 */
template <typename T>
__global__ void flashAttentionKernel(
    const T* __restrict__ Q,  // [batch, tgt_len, q_heads, head_dim]
    const T* __restrict__ K,  // [batch, src_len, kv_heads, head_dim]
    const T* __restrict__ V,  // [batch, src_len, kv_heads, head_dim]
    T* __restrict__ O,        // [batch, tgt_len, q_heads, head_dim]
    int batch_size, int tgt_len, int src_len,
    int q_heads, int kv_heads, int head_dim,
    bool is_causal, float scale) {

  // 块处理（batch_idx, tgt_pos, q_head）
  int batch_idx = blockIdx.z;
  int tgt_pos = blockIdx.y;
  int q_head = blockIdx.x;

  // 将查询头映射到键值头（用于 GQA）
  int kv_head = (q_head * kv_heads) / q_heads;

  // 共享内存布局: [q_vec | kv_vec | output_acc]
  extern __shared__ char smem_char[];
  float* q_vec = reinterpret_cast<float*>(smem_char);
  float* kv_vec = &q_vec[head_dim];
  float* output_acc = &kv_vec[head_dim];

  // 加载查询向量并初始化输出累加器
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    int q_idx = ((batch_idx * tgt_len + tgt_pos) * q_heads + q_head) * head_dim + d;
    q_vec[d] = toFloat(Q[q_idx]);
    output_acc[d] = 0.0f;
  }
  __syncthreads();

  // 共享内存中的 softmax 统计
  __shared__ float max_score;
  __shared__ float sum_exp;

  if (threadIdx.x == 0) {
    max_score = -INFINITY;
    sum_exp = 0.0f;
  }
  __syncthreads();

  // 有效的源序列长度（用于因果遮罩）
  int effective_src_len = is_causal ? min(tgt_pos + 1, src_len) : src_len;

  // 处理每个键/值位置
  for (int src_pos = 0; src_pos < effective_src_len; src_pos++) {
    // 加载键向量
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      int k_idx = ((batch_idx * src_len + src_pos) * kv_heads + kv_head) * head_dim + d;
      kv_vec[d] = toFloat(K[k_idx]);
    }
    __syncthreads();

    // 计算点积 Q·K，使用双精度以提高精度
    double thread_dot = 0.0;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      thread_dot += (double)q_vec[d] * (double)kv_vec[d];
    }

    // 使用共享内存进行归约以保持精度
    __shared__ double shared_dots[256];
    shared_dots[threadIdx.x] = thread_dot;
    __syncthreads();

    // 共享内存中的树形归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        shared_dots[threadIdx.x] += shared_dots[threadIdx.x + s];
      }
      __syncthreads();
    }

    __shared__ float score;
    if (threadIdx.x == 0) {
      // 使用双精度进行乘法以保持精度
      score = (float)(shared_dots[0] * (double)scale);
    }
    __syncthreads();

    // 在线 softmax：更新最大值和总和
    __shared__ float old_max;
    __shared__ float correction_factor;
    __shared__ float new_weight;

    if (threadIdx.x == 0) {
      old_max = max_score;
      float new_max = fmaxf(max_score, score);
      max_score = new_max;
      correction_factor = expf(old_max - new_max);
      new_weight = expf(score - new_max);
      sum_exp = sum_exp * correction_factor + new_weight;
    }
    __syncthreads();

    // 加载值向量
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      int v_idx = ((batch_idx * src_len + src_pos) * kv_heads + kv_head) * head_dim + d;
      kv_vec[d] = toFloat(V[v_idx]);
    }
    __syncthreads();

    // 更新输出累加器
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      output_acc[d] = output_acc[d] * correction_factor + kv_vec[d] * new_weight;
    }
    __syncthreads();
  }

  // 归一化并写入输出
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float normalized = output_acc[d] / (sum_exp + 1e-6f);
    int o_idx = ((batch_idx * tgt_len + tgt_pos) * q_heads + q_head) * head_dim + d;

    if (sizeof(T) == sizeof(float)) {
      O[o_idx] = *reinterpret_cast<T*>(&normalized);
    } else {
      O[o_idx] = __float2half(normalized);
    }
  }
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {

  // 计算缩放因子 (1 / sqrt(head_dim))
  float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  // 分配设备内存
  T *d_q, *d_k, *d_v, *d_o;
  size_t q_size = batch_size * target_seq_len * query_heads * head_dim;
  size_t k_size = batch_size * src_seq_len * kv_heads * head_dim;
  size_t v_size = batch_size * src_seq_len * kv_heads * head_dim;
  size_t o_size = batch_size * target_seq_len * query_heads * head_dim;

  RUNTIME_CHECK(cudaMalloc(&d_q, q_size * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_k, k_size * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_v, v_size * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_o, o_size * sizeof(T)));

  // 将输入数据拷贝到设备
  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), k_size * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), v_size * sizeof(T), cudaMemcpyHostToDevice));

  // 启动内核
  // 网格: (query_heads, target_seq_len, batch_size)
  dim3 grid(query_heads, target_seq_len, batch_size);
  int threads = 256;

  // 共享内存: Q 向量 + KV 向量 + 输出累加器（float） + 归约缓冲区（double）
  size_t shared_mem_size = 3 * head_dim * sizeof(float) + 256 * sizeof(double);

  flashAttentionKernel<<<grid, threads, shared_mem_size>>>(d_q, d_k, d_v, d_o,
      batch_size, target_seq_len, src_seq_len,
      query_heads, kv_heads, head_dim,
      is_causal, scale);

  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());

  // 将结果拷贝回主机
  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost));

  // 释放设备内存
  RUNTIME_CHECK(cudaFree(d_q));
  RUNTIME_CHECK(cudaFree(d_k));
  RUNTIME_CHECK(cudaFree(d_v));
  RUNTIME_CHECK(cudaFree(d_o));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
