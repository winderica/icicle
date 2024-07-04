extern "C" cudaError_t ${FIELD}_mul_cuda(
  ${FIELD}::scalar_t* vec_a, ${FIELD}::scalar_t* vec_b, int n, vec_ops::VecOpsConfig& config, ${FIELD}::scalar_t* result);

extern "C" cudaError_t ${FIELD}_add_cuda(
  ${FIELD}::scalar_t* vec_a, ${FIELD}::scalar_t* vec_b, int n, vec_ops::VecOpsConfig& config, ${FIELD}::scalar_t* result);

extern "C" cudaError_t ${FIELD}_accumulate_cuda(
  ${FIELD}::scalar_t* vec_a, ${FIELD}::scalar_t* vec_b, int n, vec_ops::VecOpsConfig& config);

extern "C" cudaError_t ${FIELD}_sub_cuda(
  ${FIELD}::scalar_t* vec_a, ${FIELD}::scalar_t* vec_b, int n, vec_ops::VecOpsConfig& config, ${FIELD}::scalar_t* result);

extern "C" cudaError_t ${FIELD}_mul_mat_cuda(
  ${FIELD}::scalar_t* vec_a, ${FIELD}::scalar_t* mat, int* row_ptr, int* col_idx, int n_rows, int n_cols, vec_ops::VecOpsConfig& config, ${FIELD}::scalar_t* result);

extern "C" cudaError_t ${FIELD}_prepare_matrix_cuda(
  ${FIELD}::scalar_t* mat,
  int* row_ptr,
  int* col_idx,
  int n_rows,
  device_context::DeviceContext& ctx,
  ${FIELD}::scalar_t* output_mat,
  int* output_row_ptr,
  int* output_col_idx);

extern "C" cudaError_t ${FIELD}_compute_t_cuda(
  ${FIELD}::scalar_t* mat_a,
  const int* row_ptr_a,
  const int* col_idx_a,
  ${FIELD}::scalar_t* mat_b,
  const int* row_ptr_b,
  const int* col_idx_b,
  ${FIELD}::scalar_t* mat_c,
  const int* row_ptr_c,
  const int* col_idx_c,
  ${FIELD}::scalar_t* z1,
  ${FIELD}::scalar_t* z2,
  int n_rows,
  int n_cols,
  device_context::DeviceContext& ctx,
  ${FIELD}::scalar_t* result);

extern "C" cudaError_t ${FIELD}_transpose_matrix_cuda(
  const ${FIELD}::scalar_t* input,
  uint32_t row_size,
  uint32_t column_size,
  ${FIELD}::scalar_t* output,
  device_context::DeviceContext& ctx,
  bool on_device,
  bool is_async);

extern "C" cudaError_t ${FIELD}_bit_reverse_cuda(
  const ${FIELD}::scalar_t* input, uint64_t n, vec_ops::BitReverseConfig& config, ${FIELD}::scalar_t* output);
