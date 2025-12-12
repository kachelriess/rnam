# adapted from lucidrains/minGRU-pytorch (MIT license)
# source: https://github.com/lucidrains/minGRU-pytorch/blob/9fe95d623b2a30f5cbc689e4640dc62403da0df5/minGRU_pytorch/minGRU.py
#
# incorporates the log-space parallel scan implementation from glassroom/heinsen_sequence (MIT license)
# source: https://github.com/glassroom/heinsen_sequence/blob/b747964a6fda7048558791e0d29060fe69035507/README.md
#
# license texts are available in licenses/
# corresponding papers are available in README.md
#
# last access: 2025-12-12

import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear import Linear


def heinsen_associative_scan_log(
    log_coeffs: torch.Tensor, log_values: torch.Tensor
) -> torch.Tensor:
    a_star = log_coeffs.cumsum(dim=-2)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=-2)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()


def g(x: torch.Tensor, log: bool) -> torch.Tensor:
    if log:
        return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))
    return torch.where(x >= 0, x + 0.5, x.sigmoid())


class minGRU(nn.Module):
    """
    A batched variant of the minGRU.

    Args:
        terms (int): number of independent transformations.
        dim (int): input/output dimension.
        expansion_factor (float): expansion factor for hidden dimension
            (default: 1.0).
    """

    def __init__(
        self, terms: int, dim: int, expansion_factor: float = 1.0
    ) -> None:
        super().__init__()

        assert 0 < terms
        assert 0 < dim

        dim_inner = int(dim * expansion_factor)
        assert 0 < dim_inner

        self.to_hidden_and_gate = Linear(terms, dim, dim_inner * 2, bias=False)
        self.to_out = (
            Linear(terms, dim_inner, dim, bias=False)
            if expansion_factor != 1.0
            else nn.Identity()
        )

    def forward(
        self, input: torch.Tensor, prev_hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward a tensor through the minGRU layer.

        If `seq_len == 1`, a single sequential iteration is run.
        Otherwise, the log-space parallel scan is used.

        Note:
            When `seq_len > 1` and `prev_hidden is not None`, the (pre-projection)
            `hidden` tensor will not be contiguous.
            See `rnam.Linear` for why this can be problematic.

        Args:
            input (torch.Tensor): Tensor of shape (terms, batch_size, seq_len, dim).
            prev_hidden (torch.Tensor, optional): Tensor of shape (terms, batch_size,
                1, hidden_dim), where hidden_dim = int(dim * expansion_factor).

        Returns:
            hidden (torch.Tensor): Tensor of shape (terms, batch_size, seq_len, dim).
            prev_hidden (torch.Tensor): Tensor of shape (terms, batch_size,
                1, hidden_dim), where hidden_dim = int(dim * expansion_factor).

        Examples:
            >>> layer = rnam.minGRU(4, 16, 2.0)
            >>> input = torch.randn(4, 64, 1_000, 16)
            >>> hidden, prev_hidden = layer(input)
            >>> print(hidden.size())
            torch.Size([4, 64, 1_000, 16])
            >>> print(prev_hidden.size())
            torch.Size([4, 64, 1, 32])
        """

        seq_len = input.shape[-2]
        hidden, gate = self.to_hidden_and_gate(input).chunk(2, dim=-1)

        if seq_len == 1:
            hidden = g(hidden, log=False)
            gate = gate.sigmoid()
            hidden = (
                torch.lerp(prev_hidden, hidden, gate)
                if prev_hidden is not None
                else (hidden * gate)
            )
        else:
            log_coeffs = -F.softplus(gate)

            log_z = -F.softplus(-gate)
            log_tilde_h = g(hidden, log=True)
            log_values = log_z + log_tilde_h

            if prev_hidden is not None:
                assert (prev_hidden > 0).all()
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))
                log_values = torch.cat((prev_hidden.log(), log_values), dim=-2)

            hidden = heinsen_associative_scan_log(log_coeffs, log_values)
            hidden = hidden[
                :, :, -seq_len:
            ]  # forces contiguous copy in rnam.Linear if prev_hidden is not None

        return self.to_out(hidden), hidden[:, :, -1:]
