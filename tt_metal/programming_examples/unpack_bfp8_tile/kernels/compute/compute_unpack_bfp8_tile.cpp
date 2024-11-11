// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace NAMESPACE {
void MAIN {
    for (int i = 0; i < LOOP_COUNT; i++) {
        DeviceZoneScopedN("UNPACK-BFP8-TILE");
        bool unpack_to_bf16 = get_arg_val<uint32_t>(0) > 0 ? true : false;
        auto cb_in_id = tt::CB::c_in0;
        auto cb_in1_id = tt::CB::c_in1;
        auto cb_out_id = tt::CB::c_out0;

        if (unpack_to_bf16) {
            continue;
        }

        binary_op_init_common(cb_in_id, cb_in1_id, cb_out_id);
        add_tiles_init();

        cb_wait_front(cb_in_id, 1);
        cb_reserve_back(cb_out_id, 1);

        tile_regs_acquire();  // acquire 8 tile registers
        add_tiles(cb_in_id, cb_in1_id, 0, 0, 0);
        tile_regs_commit();  // signal the packer

        tile_regs_wait();  // packer waits here
        pack_tile(0, cb_out_id);
        tile_regs_release();  // packer releases

        cb_pop_front(cb_in_id, 1);
        cb_push_back(cb_out_id, 1);
    }
}
}  // namespace NAMESPACE
