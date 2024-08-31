// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_adamw_device_operation.hpp"

namespace ttnn::operations::adamw {

// MorehAdamWDeviceOperationTest::program_factory_t MorehAdamWDeviceOperationTest::select_program_factory(
//     const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
//     bool some_condition_based_on_operation_attributes_and_or_tensor_args = true;
//     if (some_condition_based_on_operation_attributes_and_or_tensor_args) {
//         return SingleCore{};
//     }
//     return MultiCore{};
// }

// void MorehAdamWDeviceOperationTest::validate_on_program_cache_miss(
//     const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

// void MorehAdamWDeviceOperationTest::validate_on_program_cache_hit(
//     const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

// MorehAdamWDeviceOperationTest::shape_return_value_t MorehAdamWDeviceOperationTest::compute_output_shapes(
//     const operation_attributes_t&, const tensor_args_t& tensor_args) {
//     return tensor_args.input_tensor.tensor_attributes->shape;
// }

// MorehAdamWDeviceOperationTest::tensor_return_value_t MorehAdamWDeviceOperationTest::create_output_tensors(
//     const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
//     auto output_shape = compute_output_shapes(operation_attributes, tensor_args);
//     const auto& input_tensor = tensor_args.input_tensor;
//     return create_device_tensor(
//         output_shape,
//         input_tensor.tensor_attributes->dtype,
//         input_tensor.tensor_attributes->layout,
//         input_tensor.device());
// }

// std::tuple<MorehAdamWDeviceOperationTest::operation_attributes_t, MorehAdamWDeviceOperationTest::tensor_args_t>
// MorehAdamWDeviceOperationTest::invoke(const Tensor& input_tensor) {
//     return {operation_attributes_t{true, 42}, tensor_args_t{input_tensor}};
// }

MorehAdamWDeviceOperation::program_factory_t MorehAdamWDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return MultiCore{};
}

void MorehAdamWDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

void MorehAdamWDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

MorehAdamWDeviceOperation::shape_return_value_t MorehAdamWDeviceOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_tensor_shape = tensor_args.param_in.get_shape();

    return {
        input_tensor_shape,
        input_tensor_shape,
        input_tensor_shape,
        input_tensor_shape,
    };
}

MorehAdamWDeviceOperation::tensor_return_value_t MorehAdamWDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_shapes = compute_output_shapes(operation_attributes, tensor_args);
    auto dtype = tensor_args.param_in.get_dtype();
    Layout layout{Layout::TILE};
    auto device = tensor_args.param_in.device();

    tensor_return_value_t result;

    result.push_back(tensor_args.param_out);
    result.push_back(tensor_args.exp_avg_out);
    result.push_back(tensor_args.exp_avg_sq_out);

    if (tensor_args.max_exp_avg_sq_out.has_value()) {
        result.push_back(tensor_args.max_exp_avg_sq_out.value());
    } else {
        result.push_back(
            create_device_tensor(output_shapes.at(0), dtype, layout, device, operation_attributes.mem_config));
    }

    return std::move(result);
}

std::tuple<MorehAdamWDeviceOperation::operation_attributes_t, MorehAdamWDeviceOperation::tensor_args_t>
MorehAdamWDeviceOperation::invoke(
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
    const DeviceComputeKernelConfig compute_kernel_config) {
    return {
        operation_attributes_t{lr, beta1, beta2, eps, weight_decay, step, amsgrad, mem_config, compute_kernel_config},
        tensor_args_t{
            param_in,
            grad,
            exp_avg_in,
            exp_avg_sq_in,
            max_exp_avg_sq_in,
            param_out,
            exp_avg_out,
            exp_avg_sq_out,
            max_exp_avg_sq_out}};
}

}  // namespace ttnn::operations::adamw
