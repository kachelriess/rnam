# initialization adopted from torch.nn.Linear (BSD-3-Clause license)
# source: https://github.com/pytorch/pytorch/blob/d38164a545b4a4e4e0cf73ce67173f70574890b6/torch/nn/modules/linear.py#L117
#
# license texts are available in licenses/
#
# last access: 2025-12-11

import math

import torch
import torch.nn as nn


class Linear(nn.Module):
    """
    A batched affine/linear transformation.

    Args:
        terms (int): number of independent transformations.
        in_dim (int): input dimension.
        out_dim (int): output dimension.
        bias (bool): whether to compute an affine or a linear transformation
            (default: True).
    """

    def __init__(
        self, terms: int, in_dim: int, out_dim: int, bias: bool = True
    ) -> None:
        super().__init__()

        assert 0 < terms
        assert 0 < in_dim
        assert 0 < out_dim

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.weight = nn.Parameter(torch.empty(terms, in_dim, out_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(terms, 1, out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_dim)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward a tensor through the Linear layer.

        Note:
            This layer forces the input to be contiguous, which may incur
            additional memory overhead.

        Args:
            input (torch.Tensor): Tensor of shape (terms, *, in_dim).

        Returns:
            output (torch.Tensor): Tensor of shape (terms, *, out_dim).

        Examples:
            >>> layer = rnam.Linear(4, 16, 32)
            >>> input = torch.randn(4, 64, 16)
            >>> output = layer(input)
            >>> print(output.size())
            torch.Size([4, 64, 32])
        """

        assert 2 < input.ndim
        exceeds_3d = 3 < input.ndim

        shape = input.size()
        input = input.contiguous()  # expensive but necessary in some scenarios

        if exceeds_3d:
            input = input.view(shape[0], -1, shape[-1])

        if self.bias is not None:
            output = torch.baddbmm(self.bias, input, self.weight)
        else:
            output = torch.bmm(input, self.weight)

        if exceeds_3d:
            output = output.view(*shape[:-1], self.out_dim)

        return output
