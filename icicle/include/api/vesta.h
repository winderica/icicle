// WARNING: This file is auto-generated by a script.
// Any changes made to this file may be overwritten.
// Please modify the code generation script instead.
// Path to the code generation script: scripts/gen_c_api.py

#pragma once
#ifndef VESTA_API_H
#define VESTA_API_H

#include <cuda_runtime.h>
#include "gpu-utils/device_context.cuh"
#include "curves/params/vesta.cuh"
#include "msm/msm.cuh"
#include "vec_ops/vec_ops.cuh"

extern "C" cudaError_t vesta_precompute_msm_bases_cuda(
  vesta::affine_t* bases,
  int bases_size,
  int precompute_factor,
  int _c,
  bool are_bases_on_device,
  device_context::DeviceContext& ctx,
  vesta::affine_t* output_bases);

extern "C" cudaError_t vesta_msm_cuda(
  const vesta::scalar_t* scalars, const vesta::affine_t* points, int msm_size, msm::MSMConfig& config, vesta::projective_t* out);

extern "C" bool vesta_eq(vesta::projective_t* point1, vesta::projective_t* point2);

extern "C" void vesta_to_affine(vesta::projective_t* point, vesta::affine_t* point_out);

extern "C" void vesta_generate_projective_points(vesta::projective_t* points, int size);

extern "C" void vesta_generate_affine_points(vesta::affine_t* points, int size);

extern "C" cudaError_t vesta_affine_convert_montgomery(
  vesta::affine_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx);

extern "C" cudaError_t vesta_projective_convert_montgomery(
  vesta::projective_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx);

extern "C" cudaError_t vesta_mul_cuda(
  vesta::scalar_t* vec_a, vesta::scalar_t* vec_b, int n, vec_ops::VecOpsConfig& config, vesta::scalar_t* result);

extern "C" cudaError_t vesta_add_cuda(
  vesta::scalar_t* vec_a, vesta::scalar_t* vec_b, int n, vec_ops::VecOpsConfig& config, vesta::scalar_t* result);

extern "C" cudaError_t vesta_sub_cuda(
  vesta::scalar_t* vec_a, vesta::scalar_t* vec_b, int n, vec_ops::VecOpsConfig& config, vesta::scalar_t* result);

extern "C" cudaError_t vesta_transpose_matrix_cuda(
  const vesta::scalar_t* input,
  uint32_t row_size,
  uint32_t column_size,
  vesta::scalar_t* output,
  device_context::DeviceContext& ctx,
  bool on_device,
  bool is_async);

extern "C" void vesta_generate_scalars(vesta::scalar_t* scalars, int size);

extern "C" cudaError_t vesta_scalar_convert_montgomery(
  vesta::scalar_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx);

#endif