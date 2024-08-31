// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/moreh/moreh_adamw/moreh_adamw.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn::operations::adamw {

void bind_moreh_adamw_operation(py::module& module) {
    // bind_registered_operation(
    //     module,
    //     ttnn::prim::adamw,
    //     R"doc(example(input_tensor: ttnn.Tensor) -> ttnn.Tensor)doc",

    //     // Add pybind overloads for the C++ APIs that should be exposed to python
    //     // There should be no logic here, just a call to `self` with the correct arguments
    //     // The overload with `queue_id` argument will be added automatically for primitive operations
    //     // This specific function can be called from python as `ttnn.prim.example(input_tensor)` or
    //     // `ttnn.prim.example(input_tensor, queue_id=queue_id)`
    //     ttnn::pybind_overload_t{
    //         [](const decltype(ttnn::prim::adamw)& self, const ttnn::Tensor& input_tensor) -> ttnn::Tensor {
    //             return self(input_tensor);
    //         },
    //         py::arg("input_tensor")});

    bind_registered_operation(
        module,
        ttnn::adamw,
        R"doc(adamw(param_in: ttnn.Tensor) -> ttnn.Tensor)doc",
        // Add pybind overloads for the C++ APIs that should be exposed to python
        // There should be no logic here, just a call to `self` with the correct arguments
        // The overload with `queue_id` argument will be added automatically for primitive operations
        // This specific function can be called from python as `ttnn.prim.adamw(param_in)` or
        // `ttnn.prim.adamw(param_in, queue_id=queue_id)`
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::adamw)& self,
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
               const Tensor& param_out,
               const Tensor& exp_avg_out,
               const Tensor& exp_avg_sq_out,
               const std::optional<const Tensor> max_exp_avg_sq_out,
               // CHECK if memconfg, compute kernel config require
               const MemoryConfig& mem_config,
               std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
                return self(
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
            },
            py::arg("param_in"),
            py::arg("grad"),
            py::arg("exp_avg_in"),
            py::arg("exp_avg_sq_in"),
            py::arg("lr"),
            py::arg("beta1"),
            py::arg("beta2"),
            py::arg("eps"),
            py::arg("weight_decay"),
            py::arg("step"),
            py::arg("amsgrad"),

            py::arg("max_exp_avg_sq_in") = std::nullopt,
            py::arg("param_out"),
            py::arg("exp_avg_out"),
            py::arg("exp_avg_sq_out"),
            py::arg("max_exp_avg_sq_out") = std::nullopt,

            py::arg("mem_config") = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            py::arg("compute_kernel_config") = std::nullopt});
}

void py_module(py::module& module) { bind_moreh_adamw_operation(module); }

}  // namespace ttnn::operations::adamw
