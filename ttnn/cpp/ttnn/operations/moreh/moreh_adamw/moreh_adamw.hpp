
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/moreh_adamw_device_operation.hpp"

namespace ttnn::operations::adamw {

// A composite operation is an operation that calls multiple operations in sequence
// It is written using invoke and can be used to call multiple primitive and/or composite operations
struct MorehAdamw {
    // The user will be able to call this method as `Tensor output = ttnn::composite_example(input_tensor)` after the op
    // is registered
    static std::vector<Tensor> invoke(
        const Tensor& param_in,
        const Tensor& grad,
        const Tensor& exp_avg_in,
        const Tensor& exp_avg_sq_in,

        float lr,
        float beta1,
        float beta2,
        float eps,
        float weight_decay,
        uint32_t step,
        bool amsgrad,

        const std::optional<const Tensor> max_exp_avg_sq_in,
        const std::optional<const Tensor> param_out,
        const std::optional<const Tensor> exp_avg_out,
        const std::optional<const Tensor> exp_avg_sq_out,
        const std::optional<const Tensor> max_exp_avg_sq_out,
        const MemoryConfig& mem_config,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
        return ttnn::prim::adamw(
            param_in,
            grad,
            exp_avg_in,
            exp_avg_sq_in,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
            amsgrad,
            max_exp_avg_sq_in,
            param_out,
            exp_avg_out,
            exp_avg_sq_out,
            max_exp_avg_sq_out,
            mem_config,
            compute_kernel_config);
    }
};

}  // namespace ttnn::operations::adamw

namespace ttnn {
constexpr auto adamw = ttnn::register_operation<"ttnn::adamw", operations::adamw::MorehAdamw>();
}  // namespace ttnn
