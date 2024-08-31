// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "tt_dnn/op_library/compute_kernel_config.hpp"
#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::adamw {

struct MorehAdamWDeviceOperation {
    // Define the operation attributes. This is it to store all variables needed by operations that aren't tensors
    struct operation_attributes_t {
        float lr;
        float beta1;
        float beta2;
        float eps;
        float weight_decay;
        uint32_t step;
        bool amsgrad;
        const MemoryConfig mem_config;
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config;
    };

    // Define the tensor arguments. This is it to store all tensors passed in and/or out of the operation
    // Tensor arguments don't need to be just input tensors, they can be output tensors, input/output tensors, optional
    // tensors, etc.
    struct tensor_args_t {
        // This example will use a tensor that can only be used as an input
        const Tensor& param_in;
        const Tensor& grad;
        const Tensor& exp_avg_in;
        const Tensor& exp_avg_sq_in;
        const std::optional<const Tensor> max_exp_avg_sq_in;

        const Tensor& param_out;
        const Tensor& exp_avg_out;
        const Tensor& exp_avg_sq_out;
        const std::optional<const Tensor> max_exp_avg_sq_out;
    };

    // Define the return types for the shape(s) of the operation
    // Can be a single ttnn::Shape, std::optional<ttnn::Shape>, std::vector<ttnn::Shape>, std::tuple<ttnn::Shape> etc.
    using shape_return_value_t = std::vector<ttnn::Shape>;

    // Define the return types for the tensor(s) of the operation
    // Can be a single Tensor, std::optional<Tensor, ...>, std::vector<Tensor>, std::tuple<Tensor, ...> etc.
    using tensor_return_value_t = std::vector<Tensor>;

    // Note shape_return_value_t and tensor_return_value_t should follow the same pattern
    // i.e. if shape_return_value_t is a std::vector<std::optional<ttnn::Shape>> then tensor_return_value_t should be
    // std::vector<std::optional<Tensor>>

    struct MultiCore {
        // Shared variables are the variables that are shared between the create and override_runtime_arguments methods
        struct shared_variables_t {
            KernelHandle unary_reader_kernel_id;
            KernelHandle unary_writer_kernel_id;
            std::size_t num_cores;
            std::size_t num_cores_y;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<MultiCore>;

    // Mandatory methods

    // Select the program factory based on the operation attributes and tensor args
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // Validate the operation when it creates a program. Usually will have more checks
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    // Validate the operation when it reuses a program. Usually will have less checks
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // Compute the output shapes based on the operation attributes and tensor args
    static shape_return_value_t compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);

    // Create the output tensors based on the operation attributes and tensor args
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // API call to map user arguments to operation attributes and tensor args.
    // This is the only method that is called by the user
    // The user will be able to call the operation using `tensor_return_value_t output =
    // ttnn::prim::adamw(param_in)` after the op is registered Keep in mind that the the overload with
    // `queue_id` argument will be added automatically for primitive operations So, the user can also call this
    // operation using `tensor_return_value_t output = ttnn::prim::adamw(queue_id, param_in)`
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
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
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config);

    // Optional methods

    // In case the operation need a custom hash function, the following method can be implemented
    /* static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t&, const tensor_args_t&);
    */

    // In case the operation needs a custom create_op_performance_model, this method can be implemented
    /*
    static operation::OpPerformanceModel create_op_performance_model(
        const operation_attributes_t&,
        const tensor_args_t&,
        tensor_return_value_t&);
    */
};

}  // namespace ttnn::operations::adamw

// Register the operation with the ttnn::register_operation API to make it available to the user as
// ttnn::prim::adamw
namespace ttnn::prim {
constexpr auto adamw =
    ttnn::register_operation<"ttnn::prim::adamw", ttnn::operations::adamw::MorehAdamWDeviceOperation>();
}  // namespace ttnn::prim
