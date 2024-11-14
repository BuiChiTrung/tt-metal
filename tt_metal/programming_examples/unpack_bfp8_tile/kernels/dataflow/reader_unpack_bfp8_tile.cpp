// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include <cstdint>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tools/profiler/kernel_profiler.hpp"

void unpack_bfp8_to_bf16_tile(uint8_t *in_addr, uint16_t *out_addr, uint32_t unpack_elements) {
    uint32_t unpack_cnt = 0;
    // Not unpack the whole tile if unpack_elements is less than 1024
    for (uint32_t shared_exp_byte_id = 0; shared_exp_byte_id < 64 && unpack_cnt < unpack_elements;
         ++shared_exp_byte_id) {
        uint8_t shared_exp = in_addr[shared_exp_byte_id];
        for (uint32_t sign_mantissa_byte_id = 0; sign_mantissa_byte_id < 16 && unpack_cnt < unpack_elements;
             ++sign_mantissa_byte_id) {
            uint8_t sign_mantissa = in_addr[sign_mantissa_byte_id + shared_exp_byte_id * 16 + 64];
            bool sign = sign_mantissa & (1 << 7);
            uint8_t mantissa = sign_mantissa & ~(1 << 7);  // set sign bit to zero
            uint8_t exp = shared_exp;

            uint16_t bf16 = 0;
            if (mantissa != 0) {
                // Shift left mantissa until the 6th bit (hidden bit) is set
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

            *out_addr = bf16;
            out_addr++;
            unpack_cnt++;
        }
    }
}

void kernel_main() {
    DeviceZoneScopedN("UNPACK-BFP8-TILE");
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t src_dram_noc_x = get_arg_val<uint32_t>(1);
    uint32_t src_dram_noc_y = get_arg_val<uint32_t>(2);
    bool unpack_to_bf16 = get_arg_val<uint32_t>(3) > 0 ? true : false;
    bool verify_mode = get_arg_val<uint32_t>(4) == 0 ? true : false;
    uint32_t unpack_elements = get_arg_val<uint32_t>(5);

    uint64_t src_noc_addr = get_noc_addr(src_dram_noc_x, src_dram_noc_y, src_addr);

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;
    constexpr uint32_t cb_id_out0 = tt::CB::c_out0;

    cb_reserve_back(cb_id_in0, 1);

    uint32_t l1_write_addr_in = get_write_ptr(cb_id_in0);
    // Only read the data from DRAM if we are in verify mode. Otherwise, assume the data is already in SRAM.
    if (verify_mode) {
        noc_async_read(src_noc_addr, l1_write_addr_in, get_tile_size(cb_id_in0));
        noc_async_read_barrier();
    }

    if (unpack_to_bf16) {
        cb_reserve_back(cb_id_out0, 1);

        uint32_t l1_write_addr_out = get_write_ptr(cb_id_out0);
        auto in_addr = reinterpret_cast<uint8_t *>(l1_write_addr_in);
        auto out_addr = reinterpret_cast<uint16_t *>(l1_write_addr_out);
        unpack_bfp8_to_bf16_tile(in_addr, out_addr, unpack_elements);

        cb_push_back(cb_id_out0, 1);
    } else {
        // Fill cb_in1 with zeros and do add_tiles(cb_in0, cb_in1) in the compute kernel.
        cb_push_back(cb_id_in0, 1);
    }

    // Mimic the behavior in reality, wait for bfp8 tile unpacked to bf16 tile in CB. Do some pre-process works to bf16
    // tile before sending it to compute kernel.
    if (!verify_mode) {
        cb_wait_front(cb_id_out0, 1);
        // Do sth here ...
        // auto out_addr = reinterpret_cast<uint16_t *>(get_read_ptr(cb_id_out0));
        // for (int i = 0; i < 1024; ++i) {
        //     DPRINT << BF16(out_addr[i]) << " ";
        // }
        cb_pop_front(cb_id_out0, 1);
    }
}
