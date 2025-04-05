#include "fields/field_config.cuh"

using namespace field_config;

#include "utils/utils.h"
#include "vec_ops.cu"

namespace vec_ops {
  /**
   * Extern version of [Mul](@ref Mul) function with the template parameters
   * `S` and `E` being the [field](@ref scalar_t) (either scalar field of the curve given by `-DCURVE`
   * or standalone "STARK field" given by `-DFIELD`).
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t
  CONCAT_EXPAND(FIELD, mul_cuda)(scalar_t* vec_a, scalar_t* vec_b, int n, VecOpsConfig& config, scalar_t* result)
  {
    return mul<scalar_t>(vec_a, vec_b, n, config, result);
  }

  /**
   * Extern version of [Add](@ref Add) function with the template parameter
   * `E` being the [field](@ref scalar_t) (either scalar field of the curve given by `-DCURVE`
   * or standalone "STARK field" given by `-DFIELD`).
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t
  CONCAT_EXPAND(FIELD, add_cuda)(scalar_t* vec_a, scalar_t* vec_b, int n, VecOpsConfig& config, scalar_t* result)
  {
    return add<scalar_t>(vec_a, vec_b, n, config, result);
  }

  /**
   * Accumulate (as vec_a[i] += vec_b[i]) function with the template parameter
   * `E` being the [field](@ref scalar_t) (either scalar field of the curve given by `-DCURVE`
   * or standalone "STARK field" given by `-DFIELD`).
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t
  CONCAT_EXPAND(FIELD, accumulate_cuda)(scalar_t* vec_a, scalar_t* vec_b, int n, VecOpsConfig& config)
  {
    return add<scalar_t>(vec_a, vec_b, n, config, vec_a);
  }

  /**
   * Extern version of [Sub](@ref Sub) function with the template parameter
   * `E` being the [field](@ref scalar_t) (either scalar field of the curve given by `-DCURVE`
   * or standalone "STARK field" given by `-DFIELD`).
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t
  CONCAT_EXPAND(FIELD, sub_cuda)(scalar_t* vec_a, scalar_t* vec_b, int n, VecOpsConfig& config, scalar_t* result)
  {
    return sub<scalar_t>(vec_a, vec_b, n, config, result);
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, mul_mat_cuda)(
    scalar_t* vec_a,
    scalar_t* mat,
    int* row_ptr,
    int* col_idx,
    int n_rows,
    int n_cols,
    VecOpsConfig& config,
    scalar_t* result)
  {
    return mat_op<scalar_t>(vec_a, mat, row_ptr, col_idx, n_rows, n_cols, config, result);
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, prepare_matrix_cuda)(
    scalar_t* mat,
    int* row_ptr,
    int* col_idx,
    int* sparse_to_original,
    int* dense_to_original,
    int num_sparse_rows,
    int num_dense_rows,
    device_context::DeviceContext& ctx,
    HybridMatrix<scalar_t>* output)
  {
    return prepare_matrix<scalar_t>(
      mat, row_ptr, col_idx, sparse_to_original, dense_to_original, num_sparse_rows, num_dense_rows, ctx, output);
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, compute_t_cuda)(
    HybridMatrix<scalar_t>* a,
    HybridMatrix<scalar_t>* b,
    HybridMatrix<scalar_t>* c,
    scalar_t* z1_u,
    scalar_t* z1_x,
    scalar_t* z1_qw,
    scalar_t* z2_u,
    scalar_t* z2_x,
    scalar_t* z2_qw,
    scalar_t* e,
    int n_pub,
    int n_rows,
    int n_cols,
    device_context::DeviceContext& ctx,
    scalar_t* result)
  {
    return compute_t_hybrid<scalar_t>(
      a, b, c, z1_u, z1_x, z1_qw, z2_u, z2_x, z2_qw, e, n_pub, n_rows, n_cols, ctx, result);
  }

  extern "C" cudaError_t
  CONCAT_EXPAND(FIELD, update_e_cuda)(scalar_t* e, scalar_t* t, scalar_t* r, int n, device_context::DeviceContext& ctx)
  {
    return update_e<scalar_t>(e, t, r, n, ctx);
  }

  extern "C" cudaError_t
  CONCAT_EXPAND(FIELD, return_e_cuda)(scalar_t* d_e, int n, device_context::DeviceContext& ctx, scalar_t* h_e)
  {
    return return_e<scalar_t>(d_e, n, ctx, h_e);
  }

  /**
   * Extern version of transpose_batch function with the template parameter
   * `E` being the [field](@ref scalar_t) (either scalar field of the curve given by `-DCURVE`
   * or standalone "STARK field" given by `-DFIELD`).
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, transpose_matrix_cuda)(
    const scalar_t* input,
    uint32_t row_size,
    uint32_t column_size,
    scalar_t* output,
    device_context::DeviceContext& ctx,
    bool on_device,
    bool is_async)
  {
    return transpose_matrix<scalar_t>(input, output, row_size, column_size, ctx, on_device, is_async);
  }

  extern "C" cudaError_t
  CONCAT_EXPAND(FIELD, bit_reverse_cuda)(const scalar_t* input, uint64_t n, BitReverseConfig& config, scalar_t* output)
  {
    return bit_reverse<scalar_t>(input, n, config, output);
  }
} // namespace vec_ops