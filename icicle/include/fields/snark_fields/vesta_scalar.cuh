#pragma once
#ifndef VESTA_SCALAR_PARAMS_H
#define VESTA_SCALAR_PARAMS_H

#include "fields/storage.cuh"
#include "fields/field.cuh"
#include "fields/snark_fields/pallas_base.cuh"

namespace vesta {
  typedef pallas::fq_config fp_config;

  /**
   * Scalar field. Is always a prime field.
   */
  typedef Field<fp_config> scalar_t;
} // namespace vesta

#endif