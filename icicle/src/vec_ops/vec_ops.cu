#include <cuda.h>
#include <stdexcept>

#include "vec_ops/vec_ops.cuh"
#include "gpu-utils/device_context.cuh"
#include "utils/mont.cuh"

namespace vec_ops {

  namespace {

#define MAX_THREADS_PER_BLOCK 256

    template <typename E>
    __global__ void mul_kernel(const E* scalar_vec, const E* element_vec, int n, E* result)
    {
      int tid = blockDim.x * blockIdx.x + threadIdx.x;
      if (tid < n) { result[tid] = scalar_vec[tid] * element_vec[tid]; }
    }

    template <typename E>
    __global__ void mul_mat_kernel(const E* scalar_vec, const E* element_mat, const int* row_ptr, const int* col_idx, int n_rows, E* result)
    {
      int tid = blockDim.x * blockIdx.x + threadIdx.x;
      if (tid < n_rows) {
        for (int i = row_ptr[tid]; i < row_ptr[tid + 1]; i++)
        {
          result[tid] = result[tid] + scalar_vec[col_idx[i]] * element_mat[i];
        }
      }
    }

    template <typename E, typename S>
    __global__ void mul_scalar_kernel(const E* element_vec, const S scalar, int n, E* result)
    {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid < n) { result[tid] = element_vec[tid] * (scalar); }
    }

    template <typename E>
    __global__ void div_element_wise_kernel(const E* element_vec1, const E* element_vec2, int n, E* result)
    {
      // TODO:implement better based on https://eprint.iacr.org/2008/199
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid < n) { result[tid] = element_vec1[tid] * E::inverse(element_vec2[tid]); }
    }

    template <typename E>
    __global__ void add_kernel(const E* element_vec1, const E* element_vec2, int n, E* result)
    {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid < n) { result[tid] = element_vec1[tid] + element_vec2[tid]; }
    }

    template <typename E>
    __global__ void sub_kernel(const E* element_vec1, const E* element_vec2, int n, E* result)
    {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid < n) { result[tid] = element_vec1[tid] - element_vec2[tid]; }
    }

    template <typename E>
    __global__ void transpose_kernel(const E* in, E* out, uint32_t row_size, uint32_t column_size)
    {
      int tid = blockDim.x * blockIdx.x + threadIdx.x;
      if (tid >= row_size * column_size) return;
      out[(tid % row_size) * column_size + (tid / row_size)] = in[tid];
    }
  } // namespace

  template <typename E, void (*Kernel)(const E*, const E*, int, E*)>
  cudaError_t vec_op(const E* vec_a, const E* vec_b, int n, VecOpsConfig& config, E* result)
  {
    CHK_INIT_IF_RETURN();

    // Set the grid and block dimensions
    int num_threads = MAX_THREADS_PER_BLOCK;
    int num_blocks = (n + num_threads - 1) / num_threads;

    E *d_result, *d_alloc_vec_a, *d_alloc_vec_b;
    const E *d_vec_a, *d_vec_b;
    if (!config.is_a_on_device) {
      CHK_IF_RETURN(cudaMallocAsync(&d_alloc_vec_a, n * sizeof(E), config.ctx.stream));
      CHK_IF_RETURN(cudaMemcpyAsync(d_alloc_vec_a, vec_a, n * sizeof(E), cudaMemcpyHostToDevice, config.ctx.stream));
      d_vec_a = d_alloc_vec_a;
    } else {
      d_vec_a = vec_a;
    }

    if (!config.is_b_on_device) {
      CHK_IF_RETURN(cudaMallocAsync(&d_alloc_vec_b, n * sizeof(E), config.ctx.stream));
      CHK_IF_RETURN(cudaMemcpyAsync(d_alloc_vec_b, vec_b, n * sizeof(E), cudaMemcpyHostToDevice, config.ctx.stream));
      d_vec_b = d_alloc_vec_b;
    } else {
      d_vec_b = vec_b;
    }

    if (!config.is_result_on_device) {
      CHK_IF_RETURN(cudaMallocAsync(&d_result, n * sizeof(E), config.ctx.stream));
    } else {
      d_result = result;
    }

    // Call the kernel to perform element-wise operation
    Kernel<<<num_blocks, num_threads, 0, config.ctx.stream>>>(d_vec_a, d_vec_b, n, d_result);

    if (!config.is_a_on_device) { CHK_IF_RETURN(cudaFreeAsync(d_alloc_vec_a, config.ctx.stream)); }
    if (!config.is_b_on_device) { CHK_IF_RETURN(cudaFreeAsync(d_alloc_vec_b, config.ctx.stream)); }

    if (!config.is_result_on_device) {
      CHK_IF_RETURN(cudaMemcpyAsync(result, d_result, n * sizeof(E), cudaMemcpyDeviceToHost, config.ctx.stream));
      CHK_IF_RETURN(cudaFreeAsync(d_result, config.ctx.stream));
    }

    if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(config.ctx.stream));

    return CHK_LAST();
  }

  template <typename E>
  cudaError_t mat_op(const E* vec, const E* mat, const int* row_ptr, const int* col_idx, int n_rows, VecOpsConfig& config, E* result)
  {
    CHK_INIT_IF_RETURN();

    int n = row_ptr[n_rows];

    // Set the grid and block dimensions
    int num_threads = MAX_THREADS_PER_BLOCK;
    int num_blocks = (n + num_threads - 1) / num_threads;

    E *d_result, *d_alloc_vec_a, *d_alloc_mat;
    int *d_alloc_row_ptr, *d_alloc_col_idx;
    const E *d_vec_a, *d_mat;
    const int *d_row_ptr, *d_col_idx;
    if (!config.is_a_on_device) {
      CHK_IF_RETURN(cudaMallocAsync(&d_alloc_vec_a, n * sizeof(E), config.ctx.stream));
      CHK_IF_RETURN(cudaMemcpyAsync(d_alloc_vec_a, vec, n * sizeof(E), cudaMemcpyHostToDevice, config.ctx.stream));
      d_vec_a = d_alloc_vec_a;
    } else {
      d_vec_a = vec;
    }

    if (!config.is_b_on_device) {
      CHK_IF_RETURN(cudaMallocAsync(&d_alloc_mat, n * sizeof(E), config.ctx.stream));
      CHK_IF_RETURN(cudaMemcpyAsync(d_alloc_mat, mat, n * sizeof(E), cudaMemcpyHostToDevice, config.ctx.stream));
      d_mat = d_alloc_mat;
      CHK_IF_RETURN(cudaMallocAsync(&d_alloc_row_ptr, n_rows * sizeof(int), config.ctx.stream));
      CHK_IF_RETURN(cudaMemcpyAsync(d_alloc_row_ptr, row_ptr, n_rows * sizeof(int), cudaMemcpyHostToDevice, config.ctx.stream));
      d_row_ptr = d_alloc_row_ptr;
      CHK_IF_RETURN(cudaMallocAsync(&d_alloc_col_idx, n * sizeof(int), config.ctx.stream));
      CHK_IF_RETURN(cudaMemcpyAsync(d_alloc_col_idx, mat, n * sizeof(int), cudaMemcpyHostToDevice, config.ctx.stream));
      d_col_idx = d_alloc_col_idx;
    } else {
      d_mat = mat;
      d_row_ptr = row_ptr;
      d_col_idx = col_idx;
    }

    if (!config.is_result_on_device) {
      CHK_IF_RETURN(cudaMallocAsync(&d_result, n_rows * sizeof(E), config.ctx.stream));
    } else {
      d_result = result;
    }

    // Call the kernel to perform element-wise operation
    mul_mat_kernel<<<num_blocks, num_threads, 0, config.ctx.stream>>>(d_vec_a, d_mat, d_row_ptr, d_col_idx, n_rows, d_result);

    if (!config.is_a_on_device) { CHK_IF_RETURN(cudaFreeAsync(d_alloc_vec_a, config.ctx.stream)); }
    if (!config.is_b_on_device) {
      CHK_IF_RETURN(cudaFreeAsync(d_alloc_mat, config.ctx.stream));
      CHK_IF_RETURN(cudaFreeAsync(d_alloc_row_ptr, config.ctx.stream));
      CHK_IF_RETURN(cudaFreeAsync(d_alloc_col_idx, config.ctx.stream));
    }

    if (!config.is_result_on_device) {
      CHK_IF_RETURN(cudaMemcpyAsync(result, d_result, n_rows * sizeof(E), cudaMemcpyDeviceToHost, config.ctx.stream));
      CHK_IF_RETURN(cudaFreeAsync(d_result, config.ctx.stream));
    }

    if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(config.ctx.stream));

    return CHK_LAST();
  }

  template <typename E>
  cudaError_t mul(const E* vec_a, const E* vec_b, int n, VecOpsConfig& config, E* result)
  {
    return vec_op<E, mul_kernel>(vec_a, vec_b, n, config, result);
  }

  template <typename E>
  cudaError_t add(const E* vec_a, const E* vec_b, int n, VecOpsConfig& config, E* result)
  {
    return vec_op<E, add_kernel>(vec_a, vec_b, n, config, result);
  }

  template <typename E>
  cudaError_t sub(const E* vec_a, const E* vec_b, int n, VecOpsConfig& config, E* result)
  {
    return vec_op<E, sub_kernel>(vec_a, vec_b, n, config, result);
  }

  template <typename E>
  cudaError_t transpose_matrix(
    const E* mat_in,
    E* mat_out,
    uint32_t row_size,
    uint32_t column_size,
    device_context::DeviceContext& ctx,
    bool on_device,
    bool is_async)
  {
    int number_of_threads = MAX_THREADS_PER_BLOCK;
    int number_of_blocks = (row_size * column_size + number_of_threads - 1) / number_of_threads;
    cudaStream_t stream = ctx.stream;

    const E* d_mat_in;
    E* d_allocated_input = nullptr;
    E* d_mat_out;
    if (!on_device) {
      CHK_IF_RETURN(cudaMallocAsync(&d_allocated_input, row_size * column_size * sizeof(E), ctx.stream));
      CHK_IF_RETURN(cudaMemcpyAsync(
        d_allocated_input, mat_in, row_size * column_size * sizeof(E), cudaMemcpyHostToDevice, ctx.stream));

      CHK_IF_RETURN(cudaMallocAsync(&d_mat_out, row_size * column_size * sizeof(E), ctx.stream));
      d_mat_in = d_allocated_input;
    } else {
      d_mat_in = mat_in;
      d_mat_out = mat_out;
    }

    transpose_kernel<<<number_of_blocks, number_of_threads, 0, stream>>>(d_mat_in, d_mat_out, row_size, column_size);

    if (!on_device) {
      CHK_IF_RETURN(
        cudaMemcpyAsync(mat_out, d_mat_out, row_size * column_size * sizeof(E), cudaMemcpyDeviceToHost, ctx.stream));
      CHK_IF_RETURN(cudaFreeAsync(d_mat_out, ctx.stream));
      CHK_IF_RETURN(cudaFreeAsync(d_allocated_input, ctx.stream));
    }
    if (!is_async) return CHK_STICKY(cudaStreamSynchronize(ctx.stream));

    return CHK_LAST();
  }
} // namespace vec_ops