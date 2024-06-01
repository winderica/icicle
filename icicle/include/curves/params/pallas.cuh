#pragma once
#ifndef PALLAS_PARAMS_H
#define PALLAS_PARAMS_H

#include "fields/storage.cuh"

#include "curves/macro.h"
#include "curves/projective.cuh"
#include "fields/snark_fields/pallas_base.cuh"
#include "fields/snark_fields/pallas_scalar.cuh"

namespace pallas {
  // G1 generator
  static constexpr storage<fq_config::limbs_count> g1_gen_x = {0x00000000, 0x992d30ed, 0x094cf91b, 0x224698fc, 0x00000000, 0x00000000, 0x00000000, 0x40000000};
  static constexpr storage<fq_config::limbs_count> g1_gen_y = {0x00000002, 0x00000000, 0x00000000, 0x00000000,
                                                               0x00000000, 0x00000000, 0x00000000, 0x00000000};

  static constexpr storage<fq_config::limbs_count> weierstrass_b = {0x00000005, 0x00000000, 0x00000000, 0x00000000,
                                                                    0x00000000, 0x00000000, 0x00000000, 0x00000000};

  CURVE_DEFINITIONS
} // namespace pallas

#endif
