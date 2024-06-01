#pragma once
#ifndef VESTA_BASE_PARAMS_H
#define VESTA_BASE_PARAMS_H

#include "fields/storage.cuh"
#include "fields/snark_fields/pallas_scalar.cuh"

namespace vesta {
  typedef pallas::fp_config fq_config;
}

#endif