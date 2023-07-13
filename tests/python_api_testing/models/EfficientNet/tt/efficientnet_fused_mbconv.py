import torch
from typing import List

from python_api_testing.models.EfficientNet.tt.efficientnet_conv import TtEfficientnetConv2dNormActivation
from python_api_testing.models.EfficientNet.tt.efficientnet_mbconv import _MBConvConfig


class FusedMBConvConfig(_MBConvConfig):
    # Stores information listed at Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
    ) -> None:
        super().__init__(
            expand_ratio,
            kernel,
            stride,
            input_channels,
            out_channels,
            num_layers,
        )


class TtEfficientnetFusedMBConv(torch.nn.Module):
    def __init__(
        self,
        state_dict,
        base_address,
        device,
        cnf: FusedMBConvConfig,
        stochastic_depth_prob: float,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = (
            cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        )

        layers: List[torch.nn.Module] = []
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)

        if expanded_channels != cnf.input_channels:
            # fused expand
            layers.append(
                TtEfficientnetConv2dNormActivation(
                    state_dict=state_dict,
                    base_address=f"{base_address}.block.{len(layers)}",
                    device=device,
                    in_channels=cnf.input_channels,
                    out_channels=expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    activation_layer=True,
                )
            )

            # project
            layers.append(
                TtEfficientnetConv2dNormActivation(
                    state_dict=state_dict,
                    base_address=f"{base_address}.block.{len(layers)}",
                    device=device,
                    in_channels=expanded_channels,
                    out_channels=cnf.out_channels,
                    kernel_size=1,
                    activation_layer=False,
                )
            )
        else:
            layers.append(
                TtEfficientnetConv2dNormActivation(
                    state_dict=state_dict,
                    base_address=f"{base_address}.block.{len(layers)}",
                    device=device,
                    in_channels=cnf.input_channels,
                    out_channels=cnf.out_channels,
                    kernel_size=cnf.kernel,
                    activation_layer=True,
                )
            )

        self.block = torch.nn.Sequential(*layers)
        # self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, x):
        result = self.block(x)

        if self.use_res_connect:
            # result = self.stochastic_depth(result)
            result += x

        return result
