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
  int* sparse_to_original,
  int* dense_to_original,
  int num_sparse_rows,
  int num_dense_rows,
  device_context::DeviceContext& ctx,
  HybridMatrix<${FIELD}::scalar_t>* output);

extern "C" cudaError_t ${FIELD}_compute_t_cuda(
  HybridMatrix<${FIELD}::scalar_t>* a,
  HybridMatrix<${FIELD}::scalar_t>* b,
  HybridMatrix<${FIELD}::scalar_t>* c,
  ${FIELD}::scalar_t* z1_u,
  ${FIELD}::scalar_t* z1_x,
  ${FIELD}::scalar_t* z1_qw,
  ${FIELD}::scalar_t* z2_u,
  ${FIELD}::scalar_t* z2_x,
  ${FIELD}::scalar_t* z2_qw,
  ${FIELD}::scalar_t* e,
  int n_pub,
  int n_rows,
  int n_cols,
  device_context::DeviceContext& ctx,
  ${FIELD}::scalar_t* result);

extern "C" cudaError_t ${FIELD}_update_e_cuda(
  ${FIELD}::scalar_t* e,
  ${FIELD}::scalar_t* t,
  ${FIELD}::scalar_t* r,
  int n,
  device_context::DeviceContext& ctx);

extern "C" cudaError_t ${FIELD}_return_e_cuda(
  ${FIELD}::scalar_t* d_e,
  int n,
  device_context::DeviceContext& ctx,
  ${FIELD}::scalar_t* h_e);

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
