// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace NAMESPACE {
void MAIN {
    DeviceZoneScopedN("UNPACK-BFP8-TILE");
    bool unpack_to_bf16 = get_arg_val<uint32_t>(0) > 0 ? true : false;
    auto cb_id_in0 = tt::CB::c_in0;
    auto cb_id_out0 = tt::CB::c_out0;

    if (unpack_to_bf16) {
        return;
    }
    binary_op_init_common(cb_id_in0, cb_id_out0);

    cb_wait_front(cb_id_in0, 1);
    tile_regs_acquire();  // acquire 8 tile registers
    copy_tile_to_dst_init_short();
    copy_tile(cb_id_in0, 0, 0);
    tile_regs_commit();  // signal the packer
    cb_pop_front(cb_id_in0, 1);

    cb_reserve_back(cb_id_out0, 1);
    tile_regs_wait();  // packer waits here
    pack_tile(0, cb_id_out0);
    tile_regs_release();  // packer releases
    cb_push_back(cb_id_out0, 1);
}
}  // namespace NAMESPACE
