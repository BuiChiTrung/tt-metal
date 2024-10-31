// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace NAMESPACE {
void MAIN {
    for (int i = 0; i < LOOP_COUNT; i++) {
        DeviceZoneScopedN("TEST-ADD-BFP8-TILES");
        bool unpack_to_bf16 = get_arg_val<uint32_t>(0) > 0 ? true : false;
        auto cb_src0 = tt::CB::c_in0;
        auto cb_src1 = tt::CB::c_in1;
        constexpr auto cb_out0 = tt::CB::c_out0;

        if (unpack_to_bf16) {
            cb_src0 = tt::CB::c_intermed0;
            cb_src1 = tt::CB::c_intermed1;
        }

        binary_op_init_common(cb_src0, cb_src1, cb_out0);
        add_tiles_init();

        // wait for a block of tiles in each of input CBs
        cb_wait_front(cb_src0, 1);
        cb_wait_front(cb_src1, 1);

        cb_reserve_back(cb_out0, 1);

        tile_regs_acquire();  // acquire 8 tile registers

        add_tiles(cb_src0, cb_src1, 0, 0, 0);

        tile_regs_commit();  // signal the packer

        tile_regs_wait();  // packer waits here
        pack_tile(0, cb_out0);
        tile_regs_release();  // packer releases

        cb_pop_front(cb_src0, 1);
        cb_pop_front(cb_src1, 1);

        cb_push_back(cb_out0, 1);
    }
}
}  // namespace NAMESPACE
