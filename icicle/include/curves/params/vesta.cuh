#pragma once
#ifndef VESTA_PARAMS_H
#define VESTA_PARAMS_H

#include "fields/storage.cuh"

#include "curves/macro.h"
#include "curves/projective.cuh"
#include "fields/snark_fields/vesta_base.cuh"
#include "fields/snark_fields/vesta_scalar.cuh"

namespace vesta {
  typedef pallas::fp_config fq_config;

  // G1 generator
  static constexpr storage<fq_config::limbs_count> g1_gen_x = {0x00000000, 0x8c46eb21, 0x0994a8dd, 0x224698fc, 0x00000000, 0x00000000, 0x00000000, 0x40000000};
  static constexpr storage<fq_config::limbs_count> g1_gen_y = {0x00000002, 0x00000000, 0x00000000, 0x00000000,
                                                               0x00000000, 0x00000000, 0x00000000, 0x00000000};

  static constexpr storage<fq_config::limbs_count> weierstrass_b = {0x00000005, 0x00000000, 0x00000000, 0x00000000,
                                                                    0x00000000, 0x00000000, 0x00000000, 0x00000000};

  CURVE_DEFINITIONS
} // namespace vesta

#endif
