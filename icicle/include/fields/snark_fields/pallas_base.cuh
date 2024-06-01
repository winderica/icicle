#pragma once
#ifndef PALLAS_BASE_PARAMS_H
#define PALLAS_BASE_PARAMS_H

#include "fields/storage.cuh"

namespace pallas {
  struct fq_config {
    static constexpr unsigned limbs_count = 8;
    static constexpr unsigned modulus_bit_count = 255;
    static constexpr unsigned num_of_reductions = 2;
    static constexpr storage<limbs_count> modulus = {0x00000001, 0x992d30ed, 0x094cf91b, 0x224698fc, 0x00000000, 0x00000000, 0x00000000, 0x40000000};
    static constexpr storage<limbs_count> modulus_2 = {0x00000002, 0x325a61da, 0x1299f237, 0x448d31f8, 0x00000000, 0x00000000, 0x00000000, 0x80000000};
    static constexpr storage<limbs_count> modulus_4 = {0x00000004, 0x64b4c3b4, 0x2533e46e, 0x891a63f0, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<limbs_count> neg_modulus = {0xffffffff, 0x66d2cf12, 0xf6b306e4, 0xddb96703, 0xffffffff, 0xffffffff, 0xffffffff, 0xbfffffff};
    static constexpr storage<2 * limbs_count> modulus_wide = {0x00000001, 0x992d30ed, 0x094cf91b, 0x224698fc, 0x00000000, 0x00000000, 0x00000000, 0x40000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<2 * limbs_count> modulus_squared = {0x00000001, 0x325a61da, 0x9945ada0, 0x8fa19a6b, 0xbc3c95d1, 0x47797a99, 0xf7b9cb71, 0x8496d41a, 0x80000000, 0xcc969876, 0x04a67c8d, 0x11234c7e, 0x00000000, 0x00000000, 0x00000000, 0x10000000};
    static constexpr storage<2 * limbs_count> modulus_squared_2 = {0x00000002, 0x64b4c3b4, 0x328b5b40, 0x1f4334d7, 0x78792ba3, 0x8ef2f533, 0xef7396e2, 0x092da835, 0x00000001, 0x992d30ed, 0x094cf91b, 0x224698fc, 0x00000000, 0x00000000, 0x00000000, 0x20000000};
    static constexpr storage<2 * limbs_count> modulus_squared_4 = {0x00000004, 0xc9698768, 0x6516b680, 0x3e8669ae, 0xf0f25746, 0x1de5ea66, 0xdee72dc5, 0x125b506b, 0x00000002, 0x325a61da, 0x1299f237, 0x448d31f8, 0x00000000, 0x00000000, 0x00000000, 0x40000000};
    static constexpr storage<limbs_count> m = {0xfffffffc, 0x9b4b3c4b, 0xdacc1b91, 0x76e59c0f, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};
    static constexpr storage<limbs_count> one = {0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<limbs_count> zero = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<limbs_count> montgomery_r = {0xfffffffd, 0x34786d38, 0xe41914ad, 0x992c350b, 0xffffffff, 0xffffffff, 0xffffffff, 0x3fffffff};
    static constexpr storage<limbs_count> montgomery_r_inv = {0x53a769a9, 0xcf3f8e87, 0x4077fc57, 0xac9fba6a, 0xefc89a65, 0x70cb2996, 0x1e2278d5, 0x21f1c4ff};
    // nonresidue to generate the extension field
    static constexpr uint32_t nonresidue = 1;
    // true if nonresidue is negative
    static constexpr bool nonresidue_is_negative = true;
  };
} // namespace pallas

#endif
