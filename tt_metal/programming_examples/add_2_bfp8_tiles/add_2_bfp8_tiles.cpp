// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "bfloat8.hpp"
#include "tt_backend_api_types.hpp"
#include "tt_metal/common/bfloat8.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"

using namespace tt;
using namespace tt::tt_metal;

std::vector<float> cpu_add_2_bfp8_tiles(
    const std::vector<float> &fp32_in0_vec, const std::vector<float> &fp32_in1_vec) {
    std::vector<float> fp32_result_vec;
    for (int i = 0; i < fp32_in0_vec.size(); i++) {
        fp32_result_vec.push_back(fp32_in0_vec[i] + fp32_in1_vec[i]);
    }

    return fp32_result_vec;
}

std::vector<float> npu_add_2_bfp8_tiles(
    const std::vector<float> &fp32_in0_vec,
    const std::vector<float> &fp32_in1_vec,
    bool unpack_to_bf16_in_reader_kernel) {
    Device *device = CreateDevice(0);
    std::vector<uint32_t> in0_vec = pack_fp32_vec_as_bfp8_tiles(fp32_in0_vec, true, false);
    std::vector<uint32_t> in1_vec = pack_fp32_vec_as_bfp8_tiles(fp32_in1_vec, true, false);

    constexpr uint32_t bfp8_tile_size = 1088;

    /* Setup program to execute along with its buffers and kernels to use */
    CommandQueue &cq = device->command_queue();
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t bf16_tile_size = 2 * 1024;

    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = bfp8_tile_size,
        .page_size = bfp8_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    std::shared_ptr<tt::tt_metal::Buffer> in0_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> in1_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config);

    auto in0_dram_noc_coord = in0_dram_buffer->noc_coordinates();
    auto in1_dram_noc_coord = in1_dram_buffer->noc_coordinates();
    auto dst_dram_noc_coord = dst_dram_buffer->noc_coordinates();
    uint32_t in0_dram_noc_x = in0_dram_noc_coord.x;
    uint32_t in0_dram_noc_y = in0_dram_noc_coord.y;
    uint32_t in1_dram_noc_x = in1_dram_noc_coord.x;
    uint32_t in1_dram_noc_y = in1_dram_noc_coord.y;
    uint32_t dst_dram_noc_x = dst_dram_noc_coord.x;
    uint32_t dst_dram_noc_y = dst_dram_noc_coord.y;

    /* Use L1 circular buffers to set input and output buffers that the compute engine will use */
    constexpr uint32_t in0_cb_index = CB::c_in0;
    constexpr uint32_t num_input_tiles = 1;
    CircularBufferConfig cb_in0_config =
        CircularBufferConfig(num_input_tiles * bfp8_tile_size, {{in0_cb_index, tt::DataFormat::Bfp8_b}})
            .set_page_size(in0_cb_index, bfp8_tile_size);
    CBHandle cb_in0 = tt_metal::CreateCircularBuffer(program, core, cb_in0_config);

    constexpr uint32_t in1_cb_index = CB::c_in1;
    CircularBufferConfig cb_in1_config =
        CircularBufferConfig(num_input_tiles * bfp8_tile_size, {{in1_cb_index, tt::DataFormat::Bfp8_b}})
            .set_page_size(in1_cb_index, bfp8_tile_size);
    CBHandle cb_in1 = tt_metal::CreateCircularBuffer(program, core, cb_in1_config);

    constexpr uint32_t im0_cb_index = CB::c_intermed0;
    CircularBufferConfig cb_im0_config =
        CircularBufferConfig(num_input_tiles * bf16_tile_size, {{im0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(im0_cb_index, bf16_tile_size);
    CBHandle cb_im0 = tt_metal::CreateCircularBuffer(program, core, cb_im0_config);

    constexpr uint32_t im1_cb_index = CB::c_intermed1;
    CircularBufferConfig cb_im1_config =
        CircularBufferConfig(num_input_tiles * bf16_tile_size, {{im1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(im1_cb_index, bf16_tile_size);
    CBHandle cb_im1 = tt_metal::CreateCircularBuffer(program, core, cb_im1_config);

    constexpr uint32_t output_cb_index = CB::c_out0;
    constexpr uint32_t num_output_tiles = 1;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * bfp8_tile_size, {{output_cb_index, tt::DataFormat::Bfp8_b}})
            .set_page_size(output_cb_index, bfp8_tile_size);
    CBHandle cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    constexpr int kernel_loop_count = 1;
    std::map<string, string> kernel_defines = {{"LOOP_COUNT", std::to_string(kernel_loop_count)}};

    /* Specify data movement kernels for reading/writing data to/from DRAM */
    KernelHandle binary_reader_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/add_2_bfp8_tiles/kernels/dataflow/reader_binary_1_tile.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .defines = kernel_defines});

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/add_2_bfp8_tiles/kernels/dataflow/writer_1_tile.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = kernel_defines});

    /* Set the parameters that the compute kernel will use */
    std::vector<uint32_t> compute_kernel_args = {};

    /* Use the add_tiles operation in the compute kernel */
    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/add_2_bfp8_tiles/kernels/compute/add_2_tiles.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
            .defines = kernel_defines,
        });

    EnqueueWriteBuffer(cq, in0_dram_buffer, in0_vec, false);
    EnqueueWriteBuffer(cq, in1_dram_buffer, in1_vec, false);

    /* Configure program and runtime kernel arguments, then execute */
    SetRuntimeArgs(
        program,
        binary_reader_kernel_id,
        core,
        {in0_dram_buffer->address(),
         in1_dram_buffer->address(),
         in0_dram_noc_x,
         in0_dram_noc_y,
         in1_dram_noc_x,
         in1_dram_noc_y,
         unpack_to_bf16_in_reader_kernel});
    SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {unpack_to_bf16_in_reader_kernel});
    SetRuntimeArgs(program, unary_writer_kernel_id, core, {dst_dram_buffer->address(), dst_dram_noc_x, dst_dram_noc_y});

    // Benchmark in host
    double total_time = 0;
    int host_loop_count =
        PROFILER_OP_SUPPORT_COUNT * kernel_profiler::PROFILER_L1_GUARANTEED_MARKER_COUNT / kernel_loop_count;
    host_loop_count = 1;
    for (int i = 0; i < host_loop_count; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        EnqueueProgram(cq, program, true);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        // std::cout << "Iteration " << i << " took " << elapsed.count() * 1000 << " milliseconds." << std::endl;
        if (i > 0) {  // Skip the first iteration for warm-up
            total_time += elapsed.count();
        }
    }
    double avg_time = total_time / (host_loop_count - 1);
    std::cout << "Unpack to bf16 in reader kernel: " << unpack_to_bf16_in_reader_kernel << std::endl;
    std::cout << "Average time: " << avg_time * 1000 << " milliseconds." << std::endl;

    Finish(cq);
    /* Read in result into a host vector */
    std::vector<uint32_t> result_vec;
    EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);

    tt_metal::detail::DumpDeviceProfileResults(device);
    CloseDevice(device);

    std::vector<float> float_vec = unpack_bfp8_tiles_into_float_vec(result_vec, true, false);
    return float_vec;
}

std::vector<float> generate_random_float_vector(size_t size, float min_value, float max_value) {
    std::vector<float> vec(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_value, max_value);

    for (size_t i = 0; i < size; ++i) {
        vec[i] = dis(gen);
    }

    return vec;
}

int main(int argc, char **argv) {
    std::vector<float> fp32_in0_vec = generate_random_float_vector(1024, 1, 2);
    for (size_t i = 0; i < 1024; ++i) {
        std::cout << fp32_in0_vec[i] << " ";
    }
    std::vector<float> fp32_in1_vec = generate_random_float_vector(1024, 8, 32);
    std::vector<float> npu_fp32_vec;
    npu_fp32_vec = npu_add_2_bfp8_tiles(fp32_in0_vec, fp32_in1_vec, true);

    // Verify with CPU
    std::vector<float> cpu_fp32_vec = cpu_add_2_bfp8_tiles(fp32_in0_vec, fp32_in1_vec);
    bool allclose = true;
    float rtol = 1e-01;  // relative tolerance
    float atol = 1e-03;  // absolute tolerance

    for (size_t i = 0; i < cpu_fp32_vec.size(); ++i) {
        if (std::abs(cpu_fp32_vec[i] - npu_fp32_vec[i]) > (atol + rtol * std::abs(npu_fp32_vec[i]))) {
            std::cout << i << ": " << cpu_fp32_vec[i] << " != " << npu_fp32_vec[i] << std::endl;
            allclose = false;
        }
    }

    if (allclose) {
        std::cout << "CPU and NPU results are close enough." << std::endl;
    } else {
        std::cout << "CPU and NPU results differ." << std::endl;
    }
}
