// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_pybind.hpp"

#include "ttnn/operations/moreh/moreh_adam/moreh_adam_pybind.hpp"
#include "ttnn/operations/moreh/moreh_arange/moreh_arange_pybind.hpp"
#include "ttnn/operations/moreh/moreh_nll_loss_backward/moreh_nll_loss_backward_pybind.hpp"
#include "ttnn/operations/moreh/moreh_nll_loss_unreduced_backward/moreh_nll_loss_unreduced_backward_pybind.hpp"

namespace ttnn::operations::moreh {
void bind_moreh_operations(py::module &module) {
    moreh_arange::bind_moreh_arange_operation(module);
    moreh_adam::bind_moreh_adam_operation(module);
    moreh_nll_loss_unreduced_backward::bind_moreh_nll_loss_unreduced_backward_operation(module);
    moreh_nll_loss_backward::bind_moreh_nll_loss_backward_operation(module);
}
}  // namespace ttnn::operations::moreh
