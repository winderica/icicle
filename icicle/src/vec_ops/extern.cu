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
    int n_rows,
    device_context::DeviceContext& ctx,
    scalar_t* output_mat,
    int* output_row_ptr,
    int* output_col_idx)
  {
    return prepare_matrix<scalar_t>(mat, row_ptr, col_idx, n_rows, ctx, output_mat, output_row_ptr, output_col_idx);
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, compute_t_cuda)(
    scalar_t* mat_a,
    const int* row_ptr_a,
    const int* col_idx_a,
    scalar_t* mat_b,
    const int* row_ptr_b,
    const int* col_idx_b,
    scalar_t* mat_c,
    const int* row_ptr_c,
    const int* col_idx_c,
    scalar_t* z1,
    scalar_t* z2,
    int n_rows,
    int n_cols,
    device_context::DeviceContext& ctx,
    scalar_t* result)
  {
    return compute_t<scalar_t>(mat_a, row_ptr_a, col_idx_a, mat_b, row_ptr_b, col_idx_b, mat_c, row_ptr_c, col_idx_c, z1, z2, n_rows, n_cols, ctx, result);
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