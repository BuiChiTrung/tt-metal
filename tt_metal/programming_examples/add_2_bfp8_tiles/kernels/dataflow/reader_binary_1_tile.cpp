// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include <cstdint>

#include "dataflow_api.h"
// #include "debug/dprint.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    for (int i = 0; i < LOOP_COUNT; i++) {
        DeviceZoneScopedN("TEST-ADD-BFP8-TILES");
        uint32_t src0_addr = get_arg_val<uint32_t>(0);
        uint32_t src1_addr = get_arg_val<uint32_t>(1);
        uint32_t src0_dram_noc_x = get_arg_val<uint32_t>(2);
        uint32_t src0_dram_noc_y = get_arg_val<uint32_t>(3);
        uint32_t src1_dram_noc_x = get_arg_val<uint32_t>(4);
        uint32_t src1_dram_noc_y = get_arg_val<uint32_t>(5);
        bool unpack_to_bf16 = get_arg_val<uint32_t>(6) > 0 ? true : false;

        uint64_t src0_noc_addr = get_noc_addr(src0_dram_noc_x, src0_dram_noc_y, src0_addr);
        uint64_t src1_noc_addr = get_noc_addr(src1_dram_noc_x, src1_dram_noc_y, src1_addr);

        constexpr uint32_t cb_id_in0 = tt::CB::c_in0;
        constexpr uint32_t cb_id_in1 = tt::CB::c_in1;
        constexpr uint32_t cb_id_intermed0 = tt::CB::c_intermed0;
        constexpr uint32_t cb_id_intermed1 = tt::CB::c_intermed1;

        // single-tile ublocks
        uint32_t ublock_size_bytes_0 = get_tile_size(cb_id_in0);
        uint32_t ublock_size_bytes_1 = get_tile_size(cb_id_in1);

        // read ublocks from src0/src1 to CB0/CB1, then push ublocks to compute (unpacker)
        cb_reserve_back(cb_id_in0, 1);
        cb_reserve_back(cb_id_in1, 1);

        uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);

        noc_async_read(src0_noc_addr, l1_write_addr_in0, ublock_size_bytes_0);
        noc_async_read_barrier();
        noc_async_read(src1_noc_addr, l1_write_addr_in1, ublock_size_bytes_1);
        noc_async_read_barrier();

        if (unpack_to_bf16) {
            cb_reserve_back(cb_id_intermed0, 1);
            cb_reserve_back(cb_id_intermed1, 1);

            uint32_t l1_write_addr_intermed0 = get_write_ptr(cb_id_intermed0);
            uint32_t l1_write_addr_intermed1 = get_write_ptr(cb_id_intermed1);
            for (int i = 0; i <= 1; ++i) {
                uint8_t *in_addr;
                uint16_t *intermed_addr;
                if (i == 0) {
                    in_addr = reinterpret_cast<uint8_t *>(l1_write_addr_in0);
                    intermed_addr = reinterpret_cast<uint16_t *>(l1_write_addr_intermed0);
                } else {
                    in_addr = reinterpret_cast<uint8_t *>(l1_write_addr_in1);
                    intermed_addr = reinterpret_cast<uint16_t *>(l1_write_addr_intermed1);
                }

                for (uint32_t shared_exp_byte_id = 0; shared_exp_byte_id < 64; ++shared_exp_byte_id) {
                    uint8_t shared_exp = in_addr[shared_exp_byte_id];
                    for (uint32_t sign_mantissa_byte_id = 0; sign_mantissa_byte_id < 16; ++sign_mantissa_byte_id) {
                        uint8_t sign_mantissa = in_addr[sign_mantissa_byte_id + shared_exp_byte_id * 16 + 64];
                        bool sign = sign_mantissa & (1 << 7);
                        uint8_t mantissa = sign_mantissa & ~(1 << 7);  // set sign bit to zero
                        uint8_t exp = shared_exp;

                        uint16_t bf16 = 0;
                        if (mantissa != 0) {
                            while ((mantissa & (1 << 6)) == 0) {
                                mantissa <<= 1;
                                exp--;
                            }
                            // Do another shift and clear the hidden bit
                            mantissa <<= 1;
                            mantissa &= ~(1 << 7);

                            bf16 |= mantissa;
                            bf16 |= (exp << 7);
                            bf16 |= (sign << 15);
                        }

                        // DPRINT << BF16(bf16) << " ";
                        *intermed_addr = bf16;
                        intermed_addr++;
                    }
                }
            }

            cb_push_back(cb_id_intermed0, 1);
            cb_push_back(cb_id_intermed1, 1);
        } else {
            cb_push_back(cb_id_in0, 1);
            cb_push_back(cb_id_in1, 1);
        }
    }
}
