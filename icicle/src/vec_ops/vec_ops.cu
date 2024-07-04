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
      int n_rows,
      int threads_per_row,
      E* result)
    {
      int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
      int warp_id = thread_id / threads_per_row;
      int lane_id = thread_id % threads_per_row;

      int row = warp_id;

      if (row < n_rows) {
        typename E::Wide az1 = {};
        typename E::Wide az2 = {};
        typename E::Wide bz1 = {};
        typename E::Wide bz2 = {};
        typename E::Wide cz1 = {};
        typename E::Wide cz2 = {};

        for (int i = row_ptr_a[row] + lane_id; i < row_ptr_a[row + 1]; i += threads_per_row) {
          az1 = az1 + E::mul_wide(z1[col_idx_a[i]], mat_a[i]);
          az2 = az2 + E::mul_wide(z2[col_idx_a[i]], mat_a[i]);
        }
        for (int i = row_ptr_b[row] + lane_id; i < row_ptr_b[row + 1]; i += threads_per_row) {
          bz1 = bz1 + E::mul_wide(z1[col_idx_b[i]], mat_b[i]);
          bz2 = bz2 + E::mul_wide(z2[col_idx_b[i]], mat_b[i]);
        }
        for (int i = row_ptr_c[row] + lane_id; i < row_ptr_c[row + 1]; i += threads_per_row) {
          cz1 = cz1 + E::mul_wide(z1[col_idx_c[i]], mat_c[i]);
          cz2 = cz2 + E::mul_wide(z2[col_idx_c[i]], mat_c[i]);
        }

        int temp = threads_per_row / 2;
        while (temp >= 1) {
          typename E::Wide v = {};
          for (int i = 0; i < E::TLC * 2; i++) {
            v.limbs_storage.limbs[i] = __shfl_down_sync(0xFFFFFFFF, az1.limbs_storage.limbs[i], temp);
          }
          az1 = az1 + v;
          for (int i = 0; i < E::TLC * 2; i++) {
            v.limbs_storage.limbs[i] = __shfl_down_sync(0xFFFFFFFF, az2.limbs_storage.limbs[i], temp);
          }
          az2 = az2 + v;
          for (int i = 0; i < E::TLC * 2; i++) {
            v.limbs_storage.limbs[i] = __shfl_down_sync(0xFFFFFFFF, bz1.limbs_storage.limbs[i], temp);
          }
          bz1 = bz1 + v;
          for (int i = 0; i < E::TLC * 2; i++) {
            v.limbs_storage.limbs[i] = __shfl_down_sync(0xFFFFFFFF, bz2.limbs_storage.limbs[i], temp);
          }
          bz2 = bz2 + v;
          for (int i = 0; i < E::TLC * 2; i++) {
            v.limbs_storage.limbs[i] = __shfl_down_sync(0xFFFFFFFF, cz1.limbs_storage.limbs[i], temp);
          }
          cz1 = cz1 + v;
          for (int i = 0; i < E::TLC * 2; i++) {
            v.limbs_storage.limbs[i] = __shfl_down_sync(0xFFFFFFFF, cz2.limbs_storage.limbs[i], temp);
          }
          cz2 = cz2 + v;
          temp /= 2;
        }

        if (lane_id == 0) {
          result[row] = E::reduce(
            E::mul_wide(E::reduce(az1), E::reduce(bz2)) + E::mul_wide(E::reduce(az2), E::reduce(bz1)) -
            E::mul_wide(z1[0], E::reduce(cz2)) - E::mul_wide(z2[0], E::reduce(cz1)));
        }
      }
      //
      //      int tid = blockDim.x * blockIdx.x + threadIdx.x;
      //
      //      if (tid < n_rows) {
      //        clock_t start_time = clock();
      //        typename E::Wide az1 = {};
      //        typename E::Wide az2 = {};
      //        typename E::Wide bz1 = {};
      //        typename E::Wide bz2 = {};
      //        typename E::Wide cz1 = {};
      //        typename E::Wide cz2 = {};
      //
      //        for (int i = row_ptr_a[tid]; i < row_ptr_a[tid + 1]; i++) {
      //          az1 = az1 + E::mul_wide(z1[col_idx_a[i]], mat_a[i]);
      //          az2 = az2 + E::mul_wide(z2[col_idx_a[i]], mat_a[i]);
      //        }
      //        for (int i = row_ptr_b[tid]; i < row_ptr_b[tid + 1]; i++) {
      //          bz1 = bz1 + E::mul_wide(z1[col_idx_b[i]], mat_b[i]);
      //          bz2 = bz2 + E::mul_wide(z2[col_idx_b[i]], mat_b[i]);
      //        }
      //        for (int i = row_ptr_c[tid]; i < row_ptr_c[tid + 1]; i++) {
      //          cz1 = cz1 + E::mul_wide(z1[col_idx_c[i]], mat_c[i]);
      //          cz2 = cz2 + E::mul_wide(z2[col_idx_c[i]], mat_c[i]);
      //        }
      //
      //        result[tid] = E::reduce(
      //          E::mul_wide(E::reduce(az1), E::reduce(bz2)) + E::mul_wide(E::reduce(az2), E::reduce(bz1)) -
      //          E::mul_wide(z1[0], E::reduce(cz2)) - E::mul_wide(z2[0], E::reduce(cz1)));
      //        clock_t stop_time = clock();
      //        printf("%d %lld\n", tid, stop_time - start_time);
      //      }
    }
    //
    //    template <class E>
    //    __device__ typename E::Wide warp_reduce(typename E::Wide val)
    //    {
    //      for (int offset = warpSize / 2; offset > 0; offset /= 2)
    //        typename E::Wide v = {};
    //        for (int i = 0; i < E::TLC * 2; i++) {
    //          v.limbs_storage.limbs[i] = __shfl_down_sync(0xFFFFFFFF, val.limbs_storage.limbs[i], offset);
    //        }
    //        val = val + v;
    //      }
    //      return val;
    //    }
    //
    //    __device__ unsigned int prev_power_of_2(unsigned int n)
    //    {
    //      while (n & n - 1)
    //        n = n & n - 1;
    //      return n;
    //    }
    //
    //    template <typename E>
    //    __global__ void csr_adaptive_spmv_kernel (
    //      const unsigned int n_rows,
    //      const unsigned int *col_ids,
    //      const unsigned int *row_ptr,
    //      const unsigned int *row_blocks,
    //      const E *data,
    //      const E *x,
    //      typename E::Wide *y)
    //    {
    //      const unsigned int block_row_begin = row_blocks[blockIdx.x];
    //      const unsigned int block_row_end = row_blocks[blockIdx.x + 1];
    //      const unsigned int nnz = row_ptr[block_row_end] - row_ptr[block_row_begin];
    //
    //      __shared__ typename E::Wide cache[NNZ_PER_WG];
    //
    //      if (block_row_end - block_row_begin > 1)
    //      {
    //        /// CSR-Stream case
    //        const unsigned int i = threadIdx.x;
    //        const unsigned int block_data_begin = row_ptr[block_row_begin];
    //        const unsigned int thread_data_begin = block_data_begin + i;
    //
    //        if (i < nnz)
    //          cache[i] = E::mul_wide(data[thread_data_begin], x[col_ids[thread_data_begin]]);
    //        __syncthreads ();
    //
    //        const unsigned int threads_for_reduction = prev_power_of_2 (blockDim.x / (block_row_end -
    //        block_row_begin));
    //
    //        if (threads_for_reduction > 1)
    //        {
    //          /// Reduce all non zeroes of row by multiple thread
    //          const unsigned int thread_in_block = i % threads_for_reduction;
    //          const unsigned int local_row = block_row_begin + i / threads_for_reduction;
    //
    //          typename E::Wide dot = {};
    //
    //          if (local_row < block_row_end)
    //          {
    //            const unsigned int local_first_element = row_ptr[local_row] - row_ptr[block_row_begin];
    //            const unsigned int local_last_element = row_ptr[local_row + 1] - row_ptr[block_row_begin];
    //
    //            for (unsigned int local_element = local_first_element + thread_in_block;
    //                 local_element < local_last_element;
    //                 local_element += threads_for_reduction)
    //            {
    //              dot += cache[local_element];
    //            }
    //          }
    //          __syncthreads ();
    //          cache[i] = dot;
    //
    //          /// Now each row has threads_for_reduction values in cache
    //          for (int j = threads_for_reduction / 2; j > 0; j /= 2)
    //          {
    //            /// Reduce for each row
    //            __syncthreads ();
    //
    //            const bool use_result = thread_in_block < j && i + j < NNZ_PER_WG;
    //
    //            if (use_result)
    //              dot += cache[i + j];
    //            __syncthreads ();
    //
    //            if (use_result)
    //              cache[i] = dot;
    //          }
    //
    //          if (thread_in_block == 0 && local_row < block_row_end)
    //            y[local_row] = dot;
    //        }
    //        else
    //        {
    //          /// Reduce all non zeroes of row by single thread
    //          unsigned int local_row = block_row_begin + i;
    //          while (local_row < block_row_end)
    //          {
    //            typename E::Wide dot = {};
    //
    //            for (unsigned int j = row_ptr[local_row] - block_data_begin;
    //                 j < row_ptr[local_row + 1] - block_data_begin;
    //                 j++)
    //            {
    //              dot += cache[j];
    //            }
    //
    //            y[local_row] = dot;
    //            local_row += NNZ_PER_WG;
    //          }
    //        }
    //      }
    //      else
    //      {
    //        const unsigned int row = block_row_begin;
    //        const unsigned int warp_id = threadIdx.x / 32;
    //        const unsigned int lane = threadIdx.x % 32;
    //
    //        typename E::Wide dot = {};
    //
    //        if (nnz <= 64 || NNZ_PER_WG <= 32)
    //        {
    //          /// CSR-Vector case
    //          if (row < n_rows)
    //          {
    //            const unsigned int row_start = row_ptr[row];
    //            const unsigned int row_end = row_ptr[row + 1];
    //
    //            for (unsigned int element = row_start + lane; element < row_end; element += 32)
    //              dot += E::mul_wide(data[element], x[col_ids[element]]);
    //          }
    //
    //          dot = warp_reduce (dot);
    //
    //          if (lane == 0 && warp_id == 0 && row < n_rows)
    //          {
    //            y[row] = dot;
    //          }
    //        }
    //        else
    //        {
    //          /// CSR-VectorL case
    //          if (row < n_rows)
    //          {
    //            const unsigned int row_start = row_ptr[row];
    //            const unsigned int row_end = row_ptr[row + 1];
    //
    //            for (unsigned int element = row_start + threadIdx.x; element < row_end; element += blockDim.x)
    //              dot += E::mul_wide(data[element], x[col_ids[element]]);
    //          }
    //
    //          dot = warp_reduce (dot);
    //
    //          if (lane == 0)
    //            cache[warp_id] = dot;
    //          __syncthreads ();
    //
    //          if (warp_id == 0)
    //          {
    //            dot = {};
    //
    //            for (unsigned int element = lane; element < blockDim.x / 32; element += 32)
    //              dot += cache[element];
    //
    //            dot = warp_reduce (dot);
    //
    //            if (lane == 0 && row < n_rows)
    //            {
    //              y[row] = dot;
    //            }
    //          }
    //        }
    //      }
    //    }
    //
    //
    //    unsigned int
    //    fill_row_blocks (
    //      bool fill,
    //      unsigned int rows_count,
    //      const unsigned int *row_ptr,
    //      unsigned int *row_blocks
    //    )
    //    {
    //      if (fill)
    //        row_blocks[0] = 0;
    //
    //      int last_i = 0;
    //      int current_wg = 1;
    //      unsigned int nnz_sum = 0;
    //      for (int i = 1; i <= rows_count; i++)
    //      {
    //        nnz_sum = sum + row_ptr[i] - row_ptr[i - 1];
    //
    //        if (nnz_sum == NNZ_PER_WG)
    //        {
    //          last_i = i;
    //
    //          if (fill)
    //            row_blocks[current_wg] = i;
    //          current_wg++;
    //          nnz_sum = 0;
    //        }
    //        else if (nnz_sum > NNZ_PER_WG)
    //        {
    //          if (i - last_i > 1)
    //          {
    //            if (fill)
    //              row_blocks[current_wg] = i - 1;
    //            current_wg++;
    //            i--;
    //          }
    //          else
    //          {
    //            if (fill)
    //              row_blocks[current_wg] = i;
    //            current_wg++;
    //          }
    //
    //          last_i = i;
    //          nnz_sum = 0;
    //        }
    //        else if (i - last_i > NNZ_PER_WG)
    //        {
    //          last_i = i;
    //          if (fill)
    //            row_blocks[current_wg] = i;
    //          current_wg++;
    //          nnz_sum = 0;
    //        }
    //      }
    //
    //      if (fill)
    //        row_blocks[current_wg] = rows_count;
    //
    //      return current_wg;
    //    }
    //

    template <typename E, int THREADS_PER_VECTOR, int MAX_NUM_VECTORS_PER_BLOCK>
    __global__ void spmv_light_kernel(
      int* cudaRowCounter, const int* d_ptr, const int* d_cols, const E* d_val, const E* d_vector, E* d_out, int N)
    {
      clock_t start_time = clock();
      int i;
      typename E::Wide sum = {};
      int row;
      int rowStart, rowEnd;
      int laneId = threadIdx.x % THREADS_PER_VECTOR;      // lane index in the vector
      int vectorId = threadIdx.x / THREADS_PER_VECTOR;    // vector index in the thread block
      int warpLaneId = threadIdx.x & 31;                  // lane index in the warp
      int warpVectorId = warpLaneId / THREADS_PER_VECTOR; // vector index in the warp

      __shared__ volatile int space[MAX_NUM_VECTORS_PER_BLOCK][2];

      // Get the row index
      if (warpLaneId == 0) { row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR); }
      // Broadcast the value to other threads in the same warp and compute the row index of each vector
      row = __shfl_sync(0xffffffff, row, 0) + warpVectorId;

      while (row < N) {
        // Use two threads to fetch the row offset
        if (laneId < 2) { space[vectorId][laneId] = d_ptr[row + laneId]; }
        rowStart = space[vectorId][0];
        rowEnd = space[vectorId][1];

        sum = {};
        // Compute dot product
        if (THREADS_PER_VECTOR == 32) {
          // Ensure aligned memory access
          i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

          // Process the unaligned part
          if (i >= rowStart && i < rowEnd) { sum = sum + E::mul_wide(d_val[i], d_vector[d_cols[i]]); }

          // Process the aligned part
          for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
            sum = sum + E::mul_wide(d_val[i], d_vector[d_cols[i]]);
          }
        } else {
          for (i = rowStart + laneId; i < rowEnd; i += THREADS_PER_VECTOR) {
            sum = sum + E::mul_wide(d_val[i], d_vector[d_cols[i]]);
          }
        }
        // Intra-vector reduction
        for (int offset = THREADS_PER_VECTOR >> 1; offset > 0; offset >>= 1) {
          typename E::Wide v = {};
          for (int i = 0; i < E::TLC * 2; i++) {
            v.limbs_storage.limbs[i] = __shfl_down_sync(0xFFFFFFFF, sum.limbs_storage.limbs[i], offset);
          }
          sum = sum + v;
        }

        // Save the results
        if (laneId == 0) { d_out[row] = E::reduce(sum); }

        // Get a new row index
        if (warpLaneId == 0) { row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR); }
        // Broadcast the row index to the other threads in the same warp and compute the row index of each vector
        row = __shfl_sync(0xffffffff, row, 0) + warpVectorId;
      }
    }

    template <typename E>
    __global__ void
    spmv_pcsr_kernel1(const E* d_val, const E* d_vector, const int* d_cols, int d_nnz, typename E::Wide* d_v)
    {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      int icr = blockDim.x * gridDim.x;
      while (tid < d_nnz) {
        d_v[tid] = E::mul_wide(d_val[tid], d_vector[d_cols[tid]]);
        tid += icr;
      }
    }

    template <typename E>
    __global__ void spmv_pcsr_kernel2(typename E::Wide* d_v, const int* d_ptr, int N, E* d_out)
    {
      int gid = blockIdx.x * blockDim.x + threadIdx.x;
      int tid = threadIdx.x;

      __shared__ volatile int ptr_s[threadsPerBlock + 1];
      __shared__ typename E::Wide v_s[sizeSharedMemory];

      // Load ptr into the shared memory ptr_s
      ptr_s[tid] = d_ptr[gid];

      // Assign thread 0 of every block to store the pointer for the last row handled by the block into the last shared
      // memory location
      if (tid == 0) {
        if (gid + threadsPerBlock > N) {
          ptr_s[threadsPerBlock] = d_ptr[N];
        } else {
          ptr_s[threadsPerBlock] = d_ptr[gid + threadsPerBlock];
        }
      }
      __syncthreads();

      int temp = (ptr_s[threadsPerBlock] - ptr_s[0]) / threadsPerBlock + 1;
      int nlen = min(temp * threadsPerBlock, sizeSharedMemory);
      typename E::Wide sum = {};
      int maxlen = ptr_s[threadsPerBlock];
      for (int i = ptr_s[0]; i < maxlen; i += nlen) {
        int index = i + tid;
        __syncthreads();
        // Load d_v into the shared memory v_s
        for (int j = 0; j < nlen / threadsPerBlock; j++) {
          if (index < maxlen) {
            v_s[tid + j * threadsPerBlock] = d_v[index];
            index += threadsPerBlock;
          }
        }
        __syncthreads();

        // Sum up the elements for a row
        if (!(ptr_s[tid + 1] <= i || ptr_s[tid] > i + nlen - 1)) {
          int row_s = max(ptr_s[tid] - i, 0);
          int row_e = min(ptr_s[tid + 1] - i, nlen);
          for (int j = row_s; j < row_e; j++) {
            sum = sum + v_s[j];
          }
        }
      }
      // Write result
      d_out[gid] = E::reduce(sum);
    }

    template <typename E>
    __global__ void compute_t_kernel2(
      const E* az1,
      const E* az2,
      const E* bz1,
      const E* bz2,
      const E* cz1,
      const E* cz2,
      const E* z1,
      const E* z2,
      int n_rows,
      E* result)
    {
      int i = blockDim.x * blockIdx.x + threadIdx.x;

      if (i < n_rows) {
        result[i] = E::reduce(
          E::mul_wide(az1[i], bz2[i]) + E::mul_wide(az2[i], bz1[i]) - E::mul_wide(z1[0], cz2[i]) -
          E::mul_wide(z2[0], cz1[i]));
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
  cudaError_t prepare_matrix(
    const E* mat,
    const int* row_ptr,
    const int* col_idx,
    int n_rows,
    device_context::DeviceContext& ctx,
    E* output_mat,
    int* output_row_ptr,
    int* output_col_idx)
  {
    CHK_INIT_IF_RETURN();

    int n = row_ptr[n_rows];

    cudaStream_t& stream = ctx.stream;

    CHK_IF_RETURN(cudaMemcpyAsync(output_mat, mat, n * sizeof(E), cudaMemcpyHostToDevice, ctx.stream));
    mont::from_montgomery(output_mat, n, stream, output_mat);
    CHK_IF_RETURN(
      cudaMemcpyAsync(output_row_ptr, row_ptr, (n_rows + 1) * sizeof(int), cudaMemcpyHostToDevice, ctx.stream));
    CHK_IF_RETURN(cudaMemcpyAsync(output_col_idx, col_idx, n * sizeof(int), cudaMemcpyHostToDevice, ctx.stream));

    return CHK_STICKY(cudaStreamSynchronize(ctx.stream));
  }

  template <typename E>
  cudaError_t compute_t(
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
    int n_rows,
    int n_cols,
    device_context::DeviceContext& ctx,
    E* result)
  {
    CHK_INIT_IF_RETURN();

    int threads_per_row = 32;
    // Set the grid and block dimensions
    int num_threads = MAX_THREADS_PER_BLOCK;
    int num_blocks = (threads_per_row * n_rows + num_threads - 1) / num_threads;

    cudaStream_t& stream = ctx.stream;

    E *d_result, *d_z1, *d_z2;
    CHK_IF_RETURN(cudaMallocAsync(&d_z1, n_cols * sizeof(E), stream));
    CHK_IF_RETURN(cudaMemcpyAsync(d_z1, z1, n_cols * sizeof(E), cudaMemcpyHostToDevice, stream));
    mont::from_montgomery(d_z1, n_cols, stream, d_z1);
    CHK_IF_RETURN(cudaMallocAsync(&d_z2, n_cols * sizeof(E), stream));
    CHK_IF_RETURN(cudaMemcpyAsync(d_z2, z2, n_cols * sizeof(E), cudaMemcpyHostToDevice, stream));
    mont::from_montgomery(d_z2, n_cols, stream, d_z2);

    CHK_IF_RETURN(cudaMallocAsync(&d_result, n_rows * sizeof(E), stream));

    compute_t_kernel<<<num_blocks, num_threads, 0, stream>>>(
      mat_a, row_ptr_a, col_idx_a, mat_b, row_ptr_b, col_idx_b, mat_c, row_ptr_c, col_idx_c, d_z1, d_z2, n_rows,
      threads_per_row, d_result);

    CHK_IF_RETURN(cudaFreeAsync(d_z1, stream));
    CHK_IF_RETURN(cudaFreeAsync(d_z2, stream));

    mont::to_montgomery(d_result, n_rows, stream, d_result);
    CHK_IF_RETURN(cudaMemcpyAsync(result, d_result, n_rows * sizeof(E), cudaMemcpyDeviceToHost, stream));
    CHK_IF_RETURN(cudaFreeAsync(d_result, stream));

    return CHK_STICKY(cudaStreamSynchronize(stream));
  }

  template <typename E>
  cudaError_t compute_t2(
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
    int n_rows,
    int n_cols,
    device_context::DeviceContext& ctx,
    E* result)
  {
    CHK_INIT_IF_RETURN();

    int num_blocks = (n_rows + BlockDim - 1) / BlockDim;

    cudaStream_t& stream = ctx.stream;

    E *d_result, *d_z1, *d_z2, *d_az1, *d_az2, *d_bz1, *d_bz2, *d_cz1, *d_cz2;
    CHK_IF_RETURN(cudaMallocAsync(&d_z1, n_cols * sizeof(E), stream));
    CHK_IF_RETURN(cudaMemcpyAsync(d_z1, z1, n_cols * sizeof(E), cudaMemcpyHostToDevice, stream));
    CHK_IF_RETURN(cudaMallocAsync(&d_z2, n_cols * sizeof(E), stream));
    CHK_IF_RETURN(cudaMemcpyAsync(d_z2, z2, n_cols * sizeof(E), cudaMemcpyHostToDevice, stream));

    CHK_IF_RETURN(cudaMallocAsync(&d_result, n_rows * sizeof(E), stream));
    CHK_IF_RETURN(cudaMallocAsync(&d_az1, n_rows * sizeof(E), stream));
    CHK_IF_RETURN(cudaMallocAsync(&d_az2, n_rows * sizeof(E), stream));
    CHK_IF_RETURN(cudaMallocAsync(&d_bz1, n_rows * sizeof(E), stream));
    CHK_IF_RETURN(cudaMallocAsync(&d_bz2, n_rows * sizeof(E), stream));
    CHK_IF_RETURN(cudaMallocAsync(&d_cz1, n_rows * sizeof(E), stream));
    CHK_IF_RETURN(cudaMallocAsync(&d_cz2, n_rows * sizeof(E), stream));

    int* cudaRowCounter;
    CHK_IF_RETURN(cudaMalloc(&cudaRowCounter, sizeof(int)));
    CHK_IF_RETURN(cudaMemset(cudaRowCounter, 0, sizeof(int)));
    spmv_light_kernel<E, 16, MAX_NUM_THREADS_PER_BLOCK / 16>
      <<<num_blocks, BlockDim, 0, stream>>>(cudaRowCounter, row_ptr_a, col_idx_a, mat_a, d_z1, d_az1, n_rows);
    CHK_IF_RETURN(cudaMemset(cudaRowCounter, 0, sizeof(int)));
    spmv_light_kernel<E, 16, MAX_NUM_THREADS_PER_BLOCK / 16>
      <<<num_blocks, BlockDim, 0, stream>>>(cudaRowCounter, row_ptr_a, col_idx_a, mat_a, d_z2, d_az2, n_rows);
    CHK_IF_RETURN(cudaMemset(cudaRowCounter, 0, sizeof(int)));
    spmv_light_kernel<E, 16, MAX_NUM_THREADS_PER_BLOCK / 16>
      <<<num_blocks, BlockDim, 0, stream>>>(cudaRowCounter, row_ptr_b, col_idx_b, mat_b, d_z1, d_bz1, n_rows);
    CHK_IF_RETURN(cudaMemset(cudaRowCounter, 0, sizeof(int)));
    spmv_light_kernel<E, 16, MAX_NUM_THREADS_PER_BLOCK / 16>
      <<<num_blocks, BlockDim, 0, stream>>>(cudaRowCounter, row_ptr_b, col_idx_b, mat_b, d_z2, d_bz2, n_rows);
    CHK_IF_RETURN(cudaMemset(cudaRowCounter, 0, sizeof(int)));
    spmv_light_kernel<E, 16, MAX_NUM_THREADS_PER_BLOCK / 16>
      <<<num_blocks, BlockDim, 0, stream>>>(cudaRowCounter, row_ptr_c, col_idx_c, mat_c, d_z1, d_cz1, n_rows);
    CHK_IF_RETURN(cudaMemset(cudaRowCounter, 0, sizeof(int)));
    spmv_light_kernel<E, 16, MAX_NUM_THREADS_PER_BLOCK / 16>
      <<<num_blocks, BlockDim, 0, stream>>>(cudaRowCounter, row_ptr_c, col_idx_c, mat_c, d_z2, d_cz2, n_rows);

    compute_t_kernel2<<<num_blocks, BlockDim, 0, stream>>>(
      d_az1, d_az2, d_bz1, d_bz2, d_cz1, d_cz2, d_z1, d_z2, n_rows, d_result);

    CHK_IF_RETURN(cudaFreeAsync(d_az1, stream));
    CHK_IF_RETURN(cudaFreeAsync(d_az2, stream));
    CHK_IF_RETURN(cudaFreeAsync(d_bz1, stream));
    CHK_IF_RETURN(cudaFreeAsync(d_bz2, stream));
    CHK_IF_RETURN(cudaFreeAsync(d_cz1, stream));
    CHK_IF_RETURN(cudaFreeAsync(d_cz2, stream));
    CHK_IF_RETURN(cudaFreeAsync(d_z1, stream));
    CHK_IF_RETURN(cudaFreeAsync(d_z2, stream));

    CHK_IF_RETURN(cudaMemcpyAsync(result, d_result, n_rows * sizeof(E), cudaMemcpyDeviceToHost, stream));
    CHK_IF_RETURN(cudaFreeAsync(d_result, stream));

    return CHK_STICKY(cudaStreamSynchronize(stream));
  }

  template <typename E>
  cudaError_t compute_t3(
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
    int n_rows,
    int n_cols,
    device_context::DeviceContext& ctx,
    E* result)
  {
    CHK_INIT_IF_RETURN();

    int num_blocks2 = (n_rows + threadsPerBlock - 1) / threadsPerBlock;

    cudaStream_t& stream = ctx.stream;

    E *d_result, *d_z1, *d_z2, *d_az1, *d_az2, *d_bz1, *d_bz2, *d_cz1, *d_cz2;
    typename E::Wide* d_v;
    CHK_IF_RETURN(cudaMallocAsync(&d_z1, n_cols * sizeof(E), stream));
    CHK_IF_RETURN(cudaMemcpyAsync(d_z1, z1, n_cols * sizeof(E), cudaMemcpyHostToDevice, stream));
    CHK_IF_RETURN(cudaMallocAsync(&d_z2, n_cols * sizeof(E), stream));
    CHK_IF_RETURN(cudaMemcpyAsync(d_z2, z2, n_cols * sizeof(E), cudaMemcpyHostToDevice, stream));

    int n_a, n_b, n_c;
    CHK_IF_RETURN(cudaMemcpyAsync(&n_a, &row_ptr_a[n_rows], sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHK_IF_RETURN(cudaMemcpyAsync(&n_b, &row_ptr_b[n_rows], sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHK_IF_RETURN(cudaMemcpyAsync(&n_c, &row_ptr_c[n_rows], sizeof(int), cudaMemcpyDeviceToHost, stream));

    CHK_IF_RETURN(cudaMallocAsync(&d_result, n_rows * sizeof(E), stream));
    CHK_IF_RETURN(cudaMallocAsync(&d_az1, n_rows * sizeof(E), stream));
    CHK_IF_RETURN(cudaMallocAsync(&d_az2, n_rows * sizeof(E), stream));
    CHK_IF_RETURN(cudaMallocAsync(&d_bz1, n_rows * sizeof(E), stream));
    CHK_IF_RETURN(cudaMallocAsync(&d_bz2, n_rows * sizeof(E), stream));
    CHK_IF_RETURN(cudaMallocAsync(&d_cz1, n_rows * sizeof(E), stream));
    CHK_IF_RETURN(cudaMallocAsync(&d_cz2, n_rows * sizeof(E), stream));
    CHK_IF_RETURN(cudaMallocAsync(&d_v, max(max(n_a, n_b), n_c) * sizeof(typename E::Wide), stream));

    spmv_pcsr_kernel1<E>
      <<<(n_a + BlockDim - 1) / BlockDim, BlockDim>>>(mat_a, d_z1, col_idx_a, n_a, d_v);
    spmv_pcsr_kernel2<E><<<num_blocks2, threadsPerBlock>>>(d_v, row_ptr_a, n_rows, d_az1);
    spmv_pcsr_kernel1<E>
      <<<(n_a + BlockDim - 1) / BlockDim, BlockDim>>>(mat_a, d_z2, col_idx_a, n_a, d_v);
    spmv_pcsr_kernel2<E><<<num_blocks2, threadsPerBlock>>>(d_v, row_ptr_a, n_rows, d_az2);
    spmv_pcsr_kernel1<E>
      <<<(n_b + BlockDim - 1) / BlockDim, BlockDim>>>(mat_b, d_z1, col_idx_b, n_b, d_v);
    spmv_pcsr_kernel2<E><<<num_blocks2, threadsPerBlock>>>(d_v, row_ptr_b, n_rows, d_bz1);
    spmv_pcsr_kernel1<E>
      <<<(n_b + BlockDim - 1) / BlockDim, BlockDim>>>(mat_b, d_z2, col_idx_b, n_b, d_v);
    spmv_pcsr_kernel2<E><<<num_blocks2, threadsPerBlock>>>(d_v, row_ptr_b, n_rows, d_bz2);
    spmv_pcsr_kernel1<E>
      <<<(n_c + BlockDim - 1) / BlockDim, BlockDim>>>(mat_c, d_z1, col_idx_c, n_c, d_v);
    spmv_pcsr_kernel2<E><<<num_blocks2, threadsPerBlock>>>(d_v, row_ptr_c, n_rows, d_cz1);
    spmv_pcsr_kernel1<E>
      <<<(n_c + BlockDim - 1) / BlockDim, BlockDim>>>(mat_c, d_z2, col_idx_c, n_c, d_v);
    spmv_pcsr_kernel2<E><<<num_blocks2, threadsPerBlock>>>(d_v, row_ptr_c, n_rows, d_cz2);

    compute_t_kernel2<<<num_blocks2, BlockDim, 0, stream>>>(
      d_az1, d_az2, d_bz1, d_bz2, d_cz1, d_cz2, d_z1, d_z2, n_rows, d_result);

    CHK_IF_RETURN(cudaFreeAsync(d_v, stream));
    CHK_IF_RETURN(cudaFreeAsync(d_az1, stream));
    CHK_IF_RETURN(cudaFreeAsync(d_az2, stream));
    CHK_IF_RETURN(cudaFreeAsync(d_bz1, stream));
    CHK_IF_RETURN(cudaFreeAsync(d_bz2, stream));
    CHK_IF_RETURN(cudaFreeAsync(d_cz1, stream));
    CHK_IF_RETURN(cudaFreeAsync(d_cz2, stream));
    CHK_IF_RETURN(cudaFreeAsync(d_z1, stream));
    CHK_IF_RETURN(cudaFreeAsync(d_z2, stream));

    CHK_IF_RETURN(cudaMemcpyAsync(result, d_result, n_rows * sizeof(E), cudaMemcpyDeviceToHost, stream));
    CHK_IF_RETURN(cudaFreeAsync(d_result, stream));

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
