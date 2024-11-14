// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {
void MAIN {
    DeviceZoneScopedN("UNPACK-BFP8-TILE");
    bool unpack_to_bf16 = get_arg_val<uint32_t>(0) > 0 ? true : false;
    auto cb_id_in0 = tt::CB::c_in0;
    auto cb_id_in1 = tt::CB::c_in1;
    auto cb_id_out0 = tt::CB::c_out0;

    if (unpack_to_bf16) {
        return;
    }

    binary_op_init_common(cb_id_in0, cb_id_in1, cb_id_out0);
    // ckernel::copy_tile_to_cb(cb_id_in0, cb_id_out0, 0);

    // add_tiles_init();

    cb_wait_front(cb_id_in0, 1);
    tile_regs_acquire();  // acquire 8 tile registers
    // UNPACK core can only unpack bfp8 from CB to src register, not dst reg. We have to use an binary op here.
    // cb_id_in1 is filled with zeros.
    // add_tiles(cb_id_in0, cb_id_in1, 0, 0, 0);
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
