#pragma once
#ifndef PALLAS_POSEIDON_H
#define PALLAS_POSEIDON_H

namespace poseidon_constants_pallas {
  /**
   * This inner namespace contains optimized constants for running Poseidon.
   * These constants were generated using an algorithm defined at
   * https://spec.filecoin.io/algorithms/crypto/poseidon/
   * The number in the name corresponds to the arity of hash function
   * Each array contains:
   * RoundConstants | MDSMatrix | Non-sparse matrix | Sparse matrices
  */

  int partial_rounds_2 = 56;

  int partial_rounds_4 = 56;

  int partial_rounds_8 = 57;

  int partial_rounds_11 = 57;

  // TODO
  unsigned char poseidon_constants_2[] = {};
  unsigned char poseidon_constants_4[] = {};
  unsigned char poseidon_constants_8[] = {};
  unsigned char poseidon_constants_11[] = {};
} // namespace poseidon_constants
#endif
