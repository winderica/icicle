#include <cuda.h>
#include <stdexcept>

#include "vec_ops/vec_ops.cuh"
#include "gpu-utils/device_context.cuh"
#include "utils/mont.cuh"

#define BlockDim                  256
#define MAX_NUM_THREADS_PER_BLOCK 256
#define threadsPerBlock           32
#define sizeSharedMemory          512

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
    __global__ void mul_mat_kernel(
      const E* scalar_vec, const E* element_mat, const int* row_ptr, const int* col_idx, int n_rows, E* result)
    {
      int tid = blockDim.x * blockIdx.x + threadIdx.x;
      if (tid < n_rows) {
        for (int i = row_ptr[tid]; i < row_ptr[tid + 1]; i++) {
          result[tid] = result[tid] + scalar_vec[col_idx[i]] * element_mat[i];
        }
      }
    }

    template <typename E>
    __global__ void sparseRowsKernel(
      const E* __restrict__ values,
      const int* __restrict__ col_indices,
      const int* __restrict__ row_ptrs,
      const int* __restrict__ sparse_to_original,
      const E* __restrict__ z1,
      E* __restrict__ result,
      int num_sparse_rows)
    {
      int sparse_row = blockIdx.x * blockDim.x + threadIdx.x;

      if (sparse_row < num_sparse_rows) {
        typename E::Wide sum = {};

        int row = sparse_to_original[sparse_row];

        int start = row_ptrs[row];
        int end = row_ptrs[row + 1];

        // Process non-zeros for this row in chunks for better memory access
        constexpr int CHUNK_SIZE = 4;
        int i = start;

        // Process chunks of 4 elements with loop unrolling
        for (; i + CHUNK_SIZE <= end; i += CHUNK_SIZE) {
#pragma unroll
          for (int j = 0; j < CHUNK_SIZE; j++) {
            int col = col_indices[i + j];
            sum = sum + E::mul_wide(z1[col], values[i + j]);
          }
        }

        // Process remaining elements
        for (; i < end; i++) {
          int col = col_indices[i];
          sum = sum + E::mul_wide(z1[col], values[i]);
        }

        result[row] = E::reduce(sum);
      }
    }

    template <typename E>
    __global__ void denseRowsKernel(
      const E* __restrict__ values,
      const int* __restrict__ col_indices,
      const int* __restrict__ row_ptrs,
      const int* __restrict__ dense_to_original,
      const E* __restrict__ z1,
      E* __restrict__ result,
      int num_dense_rows)
    {
      // Use warp-level parallelism (32 threads) to process each row
      int warp_id = threadIdx.x / 32;
      int lane_id = threadIdx.x % 32;
      int dense_row = blockIdx.x * (blockDim.x / 32) + warp_id;

      if (dense_row < num_dense_rows) {
        int row_idx = dense_to_original[dense_row];

        int start = row_ptrs[row_idx];
        int end = row_ptrs[row_idx + 1];

        // Each thread processes elements with stride of 32 (warp size)
        typename E::Wide thread_sum = {};

        // Process elements with stride of warp size
        for (int i = start + lane_id; i < end; i += 32) {
          int col = col_indices[i];
          thread_sum = thread_sum + E::mul_wide(values[i], z1[col]);
        }

// Warp-level reduction using shuffle operations
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
          // For Int256, we'd need custom shuffle implementation
          // This is a simplified representation of what would be needed
          typename E::Wide v = {};
#pragma unroll
          for (int i = 0; i < E::TLC * 2; i++) {
            v.limbs_storage.limbs[i] = __shfl_down_sync(0xFFFFFFFF, thread_sum.limbs_storage.limbs[i], offset);
          }
          thread_sum = thread_sum + v;
        }

        // First thread in warp writes the result
        if (lane_id == 0) { result[row_idx] = E::reduce(thread_sum); }
      }
    }

    template <typename E>
    __global__ void finish_t_kernel(
      const E* __restrict__ t1,
      const E* __restrict__ t2,
      const E* __restrict__ t3,
      const E* __restrict__ z1,
      const E* __restrict__ e,
      E* __restrict__ result,
      int num_rows)
    {
      int row = blockIdx.x * blockDim.x + threadIdx.x;

      if (row < num_rows) {
        result[row] = E::reduce(E::mul_wide(t1[row], t2[row]) - E::mul_wide(z1[0], t3[row])) - e[row];
      }
    }

    template <typename E>
    __global__ void compute_t_kernel(
      const E* mat_a,
      const int* row_ptr_a,
      const int* col_idx_a,
      const E* mat_b,
      const int* row_ptr_b,
      const int* col_idx_b,
      const E* mat_c,
      const int* row_ptr_c,
      const int* col_idx_c,
      const E* z1,
      const E* z2,
      const E* e,
      int n_rows,
      int threads_per_row,
      E* result)
    {
      int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
      int warp_id = thread_id / threads_per_row;
      int lane_id = thread_id % threads_per_row;

      int row = warp_id;

      if (row < n_rows) {
        typename E::Wide az = {};
        typename E::Wide bz = {};
        typename E::Wide cz = {};

        for (int i = row_ptr_a[row] + lane_id; i < row_ptr_a[row + 1]; i += threads_per_row) {
          az = az + E::mul_wide(z1[col_idx_a[i]] + z2[col_idx_a[i]], mat_a[i]);
        }
        for (int i = row_ptr_b[row] + lane_id; i < row_ptr_b[row + 1]; i += threads_per_row) {
          bz = bz + E::mul_wide(z1[col_idx_b[i]] + z2[col_idx_b[i]], mat_b[i]);
        }
        for (int i = row_ptr_c[row] + lane_id; i < row_ptr_c[row + 1]; i += threads_per_row) {
          cz = cz + E::mul_wide(z1[col_idx_c[i]] + z2[col_idx_c[i]], mat_c[i]);
        }

        int temp = threads_per_row / 2;
        while (temp >= 1) {
          typename E::Wide v = {};
          for (int i = 0; i < E::TLC * 2; i++) {
            v.limbs_storage.limbs[i] = __shfl_down_sync(0xFFFFFFFF, az.limbs_storage.limbs[i], temp);
          }
          az = az + v;
          for (int i = 0; i < E::TLC * 2; i++) {
            v.limbs_storage.limbs[i] = __shfl_down_sync(0xFFFFFFFF, bz.limbs_storage.limbs[i], temp);
          }
          bz = bz + v;
          for (int i = 0; i < E::TLC * 2; i++) {
            v.limbs_storage.limbs[i] = __shfl_down_sync(0xFFFFFFFF, cz.limbs_storage.limbs[i], temp);
          }
          cz = cz + v;
          temp /= 2;
        }

        if (lane_id == 0) {
          result[row] =
            E::reduce(E::mul_wide(E::reduce(az), E::reduce(bz)) - E::mul_wide(z1[0] + z2[0], E::reduce(cz))) - e[row];
        }
      }
    }

    template <typename E>
    __global__ void update_e_kernel(E* e, E* t, E* r, int n)
    {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid < n) { e[tid] = e[tid] + t[tid] * r[0]; }
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

    template <typename E>
    __global__ void bit_reverse_kernel(const E* input, uint64_t n, unsigned shift, E* output)
    {
      uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
      // Handling arbitrary vector size
      if (tid < n) {
        int reversed_index = __brevll(tid) >> shift;
        output[reversed_index] = input[tid];
      }
    }

    template <typename E>
    __global__ void bit_reverse_inplace_kernel(E* input, uint64_t n, unsigned shift)
    {
      uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
      // Handling arbitrary vector size
      if (tid < n) {
        int reversed_index = __brevll(tid) >> shift;
        if (reversed_index > tid) {
          E temp = input[tid];
          input[tid] = input[reversed_index];
          input[reversed_index] = temp;
        }
      }
    }
  } // namespace

  template <typename E, void (*Kernel)(const E*, const E*, int, E*)>
  cudaError_t vec_op(E* vec_a, const E* vec_b, int n, VecOpsConfig& config, E* result)
  {
    CHK_INIT_IF_RETURN();

    bool is_in_place = vec_a == result;

    // Set the grid and block dimensions
    int num_threads = MAX_THREADS_PER_BLOCK;
    int num_blocks = (n + num_threads - 1) / num_threads;

    E *d_result, *d_alloc_vec_a, *d_alloc_vec_b;
    E* d_vec_a;
    const E* d_vec_b;
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
      if (!is_in_place) {
        CHK_IF_RETURN(cudaMallocAsync(&d_result, n * sizeof(E), config.ctx.stream));
      } else {
        d_result = d_vec_a;
      }
    } else {
      if (!is_in_place) {
        d_result = result;
      } else {
        d_result = result = d_vec_a;
      }
    }

    // Call the kernel to perform element-wise operation
    Kernel<<<num_blocks, num_threads, 0, config.ctx.stream>>>(d_vec_a, d_vec_b, n, d_result);

    if (!config.is_result_on_device) {
      CHK_IF_RETURN(cudaMemcpyAsync(result, d_result, n * sizeof(E), cudaMemcpyDeviceToHost, config.ctx.stream));
      CHK_IF_RETURN(cudaFreeAsync(d_result, config.ctx.stream));
    }

    if (!config.is_a_on_device && !is_in_place) { CHK_IF_RETURN(cudaFreeAsync(d_alloc_vec_a, config.ctx.stream)); }
    if (!config.is_b_on_device) { CHK_IF_RETURN(cudaFreeAsync(d_alloc_vec_b, config.ctx.stream)); }

    if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(config.ctx.stream));

    return CHK_LAST();
  }

  template <typename E>
  cudaError_t mat_op(
    const E* vec,
    const E* mat,
    const int* row_ptr,
    const int* col_idx,
    int n_rows,
    int n_cols,
    VecOpsConfig& config,
    E* result)
  {
    CHK_INIT_IF_RETURN();

    int n = row_ptr[n_rows];

    // Set the grid and block dimensions
    int num_threads = MAX_THREADS_PER_BLOCK;
    int num_blocks = (n_rows + num_threads - 1) / num_threads;

    E *d_result, *d_alloc_vec_a, *d_alloc_mat;
    int *d_alloc_row_ptr, *d_alloc_col_idx;
    const E *d_vec_a, *d_mat;
    const int *d_row_ptr, *d_col_idx;
    if (!config.is_a_on_device) {
      CHK_IF_RETURN(cudaMallocAsync(&d_alloc_vec_a, n_cols * sizeof(E), config.ctx.stream));
      CHK_IF_RETURN(cudaMemcpyAsync(d_alloc_vec_a, vec, n_cols * sizeof(E), cudaMemcpyHostToDevice, config.ctx.stream));
      d_vec_a = d_alloc_vec_a;
    } else {
      d_vec_a = vec;
    }

    if (!config.is_b_on_device) {
      CHK_IF_RETURN(cudaMallocAsync(&d_alloc_mat, n * sizeof(E), config.ctx.stream));
      CHK_IF_RETURN(cudaMemcpyAsync(d_alloc_mat, mat, n * sizeof(E), cudaMemcpyHostToDevice, config.ctx.stream));
      d_mat = d_alloc_mat;
      CHK_IF_RETURN(cudaMallocAsync(&d_alloc_row_ptr, (n_rows + 1) * sizeof(int), config.ctx.stream));
      CHK_IF_RETURN(cudaMemcpyAsync(
        d_alloc_row_ptr, row_ptr, (n_rows + 1) * sizeof(int), cudaMemcpyHostToDevice, config.ctx.stream));
      d_row_ptr = d_alloc_row_ptr;
      CHK_IF_RETURN(cudaMallocAsync(&d_alloc_col_idx, n * sizeof(int), config.ctx.stream));
      CHK_IF_RETURN(
        cudaMemcpyAsync(d_alloc_col_idx, col_idx, n * sizeof(int), cudaMemcpyHostToDevice, config.ctx.stream));
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
    mul_mat_kernel<<<num_blocks, num_threads, 0, config.ctx.stream>>>(
      d_vec_a, d_mat, d_row_ptr, d_col_idx, n_rows, d_result);

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
  struct HybridMatrix {
    // Sparse portion (CSR format)
    E* values;        // Non-zero values
    int* col_indices; // Column indices for non-zeros
    int* row_ptrs;    // Row pointers into values/col_indices arrays

    int* sparse_to_original;
    int* dense_to_original;

    E* temp;

    // Matrix dimensions
    int num_sparse_rows;
    int num_dense_rows;
  };

  template <typename E>
  cudaError_t prepare_matrix(
    const E* mat,
    const int* row_ptr,
    const int* col_idx,
    const int* sparse_to_original,
    const int* dense_to_original,
    int num_sparse_rows,
    int num_dense_rows,
    device_context::DeviceContext& ctx,
    HybridMatrix<E>* output)
  {
    CHK_INIT_IF_RETURN();

    int n_rows = num_sparse_rows + num_dense_rows;

    int n = row_ptr[n_rows];

    cudaStream_t& stream = ctx.stream;

    E *d_mat, *d_temp;
    int *d_row_ptr, *d_col_idx, *d_sparse_to_original, *d_dense_to_original;
    CHK_IF_RETURN(cudaMallocAsync(&d_mat, n * sizeof(E), stream));
    CHK_IF_RETURN(cudaMemcpyAsync(d_mat, mat, n * sizeof(E), cudaMemcpyHostToDevice, stream));
    mont::from_montgomery(d_mat, n, stream, d_mat);

    CHK_IF_RETURN(cudaMallocAsync(&d_row_ptr, (n_rows + 1) * sizeof(int), stream));
    CHK_IF_RETURN(cudaMemcpyAsync(d_row_ptr, row_ptr, (n_rows + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));

    CHK_IF_RETURN(cudaMallocAsync(&d_col_idx, n * sizeof(int), stream));
    CHK_IF_RETURN(cudaMemcpyAsync(d_col_idx, col_idx, n * sizeof(int), cudaMemcpyHostToDevice, stream));

    CHK_IF_RETURN(cudaMallocAsync(&d_sparse_to_original, num_sparse_rows * sizeof(int), stream));
    CHK_IF_RETURN(cudaMemcpyAsync(
      d_sparse_to_original, sparse_to_original, num_sparse_rows * sizeof(int), cudaMemcpyHostToDevice, stream));

    CHK_IF_RETURN(cudaMallocAsync(&d_dense_to_original, num_dense_rows * sizeof(int), stream));
    CHK_IF_RETURN(cudaMemcpyAsync(
      d_dense_to_original, dense_to_original, num_dense_rows * sizeof(int), cudaMemcpyHostToDevice, stream));

    CHK_IF_RETURN(cudaMallocAsync(&d_temp, n_rows * sizeof(E), stream));

    output->values = d_mat;
    output->col_indices = d_col_idx;
    output->row_ptrs = d_row_ptr;
    output->sparse_to_original = d_sparse_to_original;
    output->dense_to_original = d_dense_to_original;
    output->num_sparse_rows = num_sparse_rows;
    output->num_dense_rows = num_dense_rows;
    output->temp = d_temp;

    return CHK_STICKY(cudaStreamSynchronize(stream));
  }

  template <typename E>
  cudaError_t compute_t_hybrid(
    const HybridMatrix<E>* a,
    const HybridMatrix<E>* b,
    const HybridMatrix<E>* c,
    const E* z1_u,
    const E* z1_x,
    const E* z1_qw,
    const E* z2_u,
    const E* z2_x,
    const E* z2_qw,
    const E* e,
    int n_pub,
    int n_rows,
    int n_cols,
    device_context::DeviceContext& ctx,
    E* result)
  {
    CHK_INIT_IF_RETURN();

    // Set the grid and block dimensions
    int num_threads = MAX_THREADS_PER_BLOCK;

    int warps_per_block = num_threads / 32;

    cudaEvent_t dataReadyEvent, a1ReadyEvent, a2ReadyEvent, b1ReadyEvent, b2ReadyEvent, c1ReadyEvent, c2ReadyEvent;
    cudaEventCreate(&dataReadyEvent);
    cudaEventCreate(&a1ReadyEvent);
    cudaEventCreate(&a2ReadyEvent);
    cudaEventCreate(&b1ReadyEvent);
    cudaEventCreate(&b2ReadyEvent);
    cudaEventCreate(&c1ReadyEvent);
    cudaEventCreate(&c2ReadyEvent);

    cudaStream_t& stream = ctx.stream;
    cudaStream_t stream1, stream2, stream3, stream4, stream5, stream6;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);
    cudaStreamCreate(&stream5);
    cudaStreamCreate(&stream6);

    E *d_z1, *d_z2;
    CHK_IF_RETURN(cudaMallocAsync(&d_z1, n_cols * sizeof(E), stream));
    CHK_IF_RETURN(cudaMemcpyAsync(d_z1, z1_u, sizeof(E), cudaMemcpyHostToDevice, stream));
    CHK_IF_RETURN(cudaMemcpyAsync(&d_z1[1], z1_x, n_pub * sizeof(E), cudaMemcpyHostToDevice, stream));
    CHK_IF_RETURN(
      cudaMemcpyAsync(&d_z1[1 + n_pub], z1_qw, (n_cols - 1 - n_pub) * sizeof(E), cudaMemcpyHostToDevice, stream));
    CHK_IF_RETURN(cudaMallocAsync(&d_z2, n_cols * sizeof(E), stream));
    CHK_IF_RETURN(cudaMemcpyAsync(d_z2, z2_u, sizeof(E), cudaMemcpyHostToDevice, stream));
    CHK_IF_RETURN(cudaMemcpyAsync(&d_z2[1], z2_x, n_pub * sizeof(E), cudaMemcpyHostToDevice, stream));
    CHK_IF_RETURN(
      cudaMemcpyAsync(&d_z2[1 + n_pub], z2_qw, (n_cols - 1 - n_pub) * sizeof(E), cudaMemcpyHostToDevice, stream));

    add_kernel<<<(n_cols + num_threads - 1) / num_threads, num_threads, 0, stream>>>(d_z1, d_z2, n_cols, d_z1);
    mont::from_montgomery(d_z1, n_cols, stream, d_z1);

    cudaEventRecord(dataReadyEvent, stream);

    cudaStreamWaitEvent(stream1, dataReadyEvent, 0);
    cudaStreamWaitEvent(stream2, dataReadyEvent, 0);
    cudaStreamWaitEvent(stream3, dataReadyEvent, 0);
    cudaStreamWaitEvent(stream4, dataReadyEvent, 0);
    cudaStreamWaitEvent(stream5, dataReadyEvent, 0);
    cudaStreamWaitEvent(stream6, dataReadyEvent, 0);

    sparseRowsKernel<<<(a->num_sparse_rows + num_threads - 1) / num_threads, num_threads, 0, stream1>>>(
      a->values, a->col_indices, a->row_ptrs, a->sparse_to_original, d_z1, a->temp, a->num_sparse_rows);
    cudaEventRecord(a1ReadyEvent, stream1);

    if (a->num_dense_rows > 0) {
      denseRowsKernel<<<(a->num_dense_rows + warps_per_block - 1) / warps_per_block, num_threads, 0, stream2>>>(
        a->values, a->col_indices, a->row_ptrs, a->dense_to_original, d_z1, a->temp, a->num_dense_rows);
    }
    cudaEventRecord(a2ReadyEvent, stream2);

    sparseRowsKernel<<<(b->num_sparse_rows + num_threads - 1) / num_threads, num_threads, 0, stream3>>>(
      b->values, b->col_indices, b->row_ptrs, b->sparse_to_original, d_z1, b->temp, b->num_sparse_rows);
    cudaEventRecord(b1ReadyEvent, stream3);

    if (b->num_dense_rows > 0) {
      denseRowsKernel<<<(b->num_dense_rows + warps_per_block - 1) / warps_per_block, num_threads, 0, stream4>>>(
        b->values, b->col_indices, b->row_ptrs, b->dense_to_original, d_z1, b->temp, b->num_dense_rows);
    }

    cudaEventRecord(b2ReadyEvent, stream4);

    sparseRowsKernel<<<(c->num_sparse_rows + num_threads - 1) / num_threads, num_threads, 0, stream5>>>(
      c->values, c->col_indices, c->row_ptrs, c->sparse_to_original, d_z1, c->temp, c->num_sparse_rows);
    cudaEventRecord(c1ReadyEvent, stream5);

    if (c->num_dense_rows > 0) {
      denseRowsKernel<<<(c->num_dense_rows + warps_per_block - 1) / warps_per_block, num_threads, 0, stream6>>>(
        c->values, c->col_indices, c->row_ptrs, c->dense_to_original, d_z1, c->temp, c->num_dense_rows);
    }
    cudaEventRecord(c2ReadyEvent, stream6);

    cudaStreamWaitEvent(stream, a1ReadyEvent, 0);
    cudaStreamWaitEvent(stream, a2ReadyEvent, 0);
    cudaStreamWaitEvent(stream, b1ReadyEvent, 0);
    cudaStreamWaitEvent(stream, b2ReadyEvent, 0);
    cudaStreamWaitEvent(stream, c1ReadyEvent, 0);
    cudaStreamWaitEvent(stream, c2ReadyEvent, 0);

    finish_t_kernel<<<(n_rows + num_threads - 1) / num_threads, num_threads, 0, stream>>>(
      a->temp, b->temp, c->temp, d_z1, e, result, n_rows);

    CHK_IF_RETURN(cudaFreeAsync(d_z1, stream));
    CHK_IF_RETURN(cudaFreeAsync(d_z2, stream));

    cudaEventDestroy(dataReadyEvent);
    cudaEventDestroy(a1ReadyEvent);
    cudaEventDestroy(a2ReadyEvent);
    cudaEventDestroy(b1ReadyEvent);
    cudaEventDestroy(b2ReadyEvent);
    cudaEventDestroy(c1ReadyEvent);
    cudaEventDestroy(c2ReadyEvent);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    cudaStreamDestroy(stream4);
    cudaStreamDestroy(stream5);
    cudaStreamDestroy(stream6);

    return CHK_LAST();
  }

  template <typename E>
  cudaError_t update_e(E* e, E* t, E* r, int n, device_context::DeviceContext& ctx)
  {
    CHK_INIT_IF_RETURN();

    int num_threads = MAX_THREADS_PER_BLOCK;
    int num_blocks = (n + num_threads - 1) / num_threads;
    cudaStream_t& stream = ctx.stream;

    E* d_r;
    CHK_IF_RETURN(cudaMallocAsync(&d_r, 1 * sizeof(E), stream));
    CHK_IF_RETURN(cudaMemcpyAsync(d_r, r, 1 * sizeof(E), cudaMemcpyHostToDevice, stream));
    mont::from_montgomery(d_r, 1, stream, d_r);

    update_e_kernel<<<num_blocks, num_threads, 0, stream>>>(e, t, d_r, n);

    CHK_IF_RETURN(cudaFreeAsync(d_r, stream));

    return CHK_LAST();
  }

  template <typename E>
  cudaError_t return_e(E* d_e, int n, device_context::DeviceContext& ctx, E* h_e)
  {
    CHK_INIT_IF_RETURN();

    cudaStream_t& stream = ctx.stream;

    E* d_f;
    CHK_IF_RETURN(cudaMallocAsync(&d_f, n * sizeof(E), stream));

    mont::to_montgomery(d_e, n, stream, d_f);
    CHK_IF_RETURN(cudaMemcpyAsync(h_e, d_f, n * sizeof(E), cudaMemcpyDeviceToHost, stream));
    //    CHK_IF_RETURN(cudaFreeAsync(d_e, stream));

    CHK_IF_RETURN(cudaFreeAsync(d_f, stream));

    return CHK_STICKY(cudaStreamSynchronize(stream));
  }

  template <typename E>
  cudaError_t mul(E* vec_a, const E* vec_b, int n, VecOpsConfig& config, E* result)
  {
    return vec_op<E, mul_kernel>(vec_a, vec_b, n, config, result);
  }

  template <typename E>
  cudaError_t add(E* vec_a, const E* vec_b, int n, VecOpsConfig& config, E* result)
  {
    return vec_op<E, add_kernel>(vec_a, vec_b, n, config, result);
  }

  template <typename E>
  cudaError_t sub(E* vec_a, const E* vec_b, int n, VecOpsConfig& config, E* result)
  {
    return vec_op<E, sub_kernel>(vec_a, vec_b, n, config, result);
  }

  template <typename E>
  cudaError_t transpose_matrix(
    const E* mat_in,
    E* mat_out,
    uint32_t row_size,
    uint32_t column_size,
    const device_context::DeviceContext& ctx,
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

  template <typename E>
  cudaError_t bit_reverse(const E* input, uint64_t size, BitReverseConfig& cfg, E* output)
  {
    if (size & (size - 1)) THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "bit_reverse: size must be a power of 2");
    if ((input == output) & (cfg.is_input_on_device != cfg.is_output_on_device))
      THROW_ICICLE_ERR(
        IcicleError_t::InvalidArgument, "bit_reverse: equal devices should have same is_on_device parameters");

    E* d_output;
    if (cfg.is_output_on_device) {
      d_output = output;
    } else {
      // allocate output on gpu
      CHK_IF_RETURN(cudaMallocAsync(&d_output, sizeof(E) * size, cfg.ctx.stream));
    }

    uint64_t shift = __builtin_clzll(size) + 1;
    uint64_t num_blocks = (size + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

    if ((input != output) & cfg.is_input_on_device) {
      bit_reverse_kernel<<<num_blocks, MAX_THREADS_PER_BLOCK, 0, cfg.ctx.stream>>>(input, size, shift, d_output);
    } else {
      if (!cfg.is_input_on_device) {
        CHK_IF_RETURN(cudaMemcpyAsync(d_output, input, sizeof(E) * size, cudaMemcpyHostToDevice, cfg.ctx.stream));
      }
      bit_reverse_inplace_kernel<<<num_blocks, MAX_THREADS_PER_BLOCK, 0, cfg.ctx.stream>>>(d_output, size, shift);
    }
    if (!cfg.is_output_on_device) {
      CHK_IF_RETURN(cudaMemcpyAsync(output, d_output, sizeof(E) * size, cudaMemcpyDeviceToHost, cfg.ctx.stream));
      CHK_IF_RETURN(cudaFreeAsync(d_output, cfg.ctx.stream));
    }
    if (!cfg.is_async) CHK_IF_RETURN(cudaStreamSynchronize(cfg.ctx.stream));
    return CHK_LAST();
  }
} // namespace vec_ops
