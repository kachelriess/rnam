import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rnam import Linear
from tests import utils


def yield_x(factory_kwargs):
    inner_dims = []
    for _ in range(3):
        inner_dims.append(3)
        yield torch.rand(2, *inner_dims, 4, **factory_kwargs)


def loop_linear(x, w, b):
    return torch.stack(
        [F.linear(x[0], w[0].T, b[0]), F.linear(x[1], w[1].T, b[1])]
    )


@pytest.mark.parametrize("factory_kwargs", utils.get_device_dtype(), ids=str)
def test_affine_forward(factory_kwargs):
    w = torch.rand(2, 4, 5, **factory_kwargs)
    b = torch.rand(2, 1, 5, **factory_kwargs)

    layer = Linear(2, 4, 5)
    layer.weight = nn.Parameter(w)
    layer.bias = nn.Parameter(b)

    for x in yield_x(factory_kwargs):
        utils.assert_allclose(layer(x), loop_linear(x, w, b))


@pytest.mark.parametrize("factory_kwargs", utils.get_device_dtype(), ids=str)
def test_linear_forward(factory_kwargs):
    w = torch.rand(2, 4, 5, **factory_kwargs)

    layer = Linear(2, 4, 5, bias=False)
    layer.weight = nn.Parameter(w)

    for x in yield_x(factory_kwargs):
        utils.assert_allclose(layer(x), loop_linear(x, w, [None, None]))


def yield_x_and_dy(factory_kwargs):
    for x in yield_x(factory_kwargs):
        yield x, torch.rand(2, *x.size()[1:-1], 5, **factory_kwargs)


@pytest.mark.parametrize("factory_kwargs", utils.get_device_dtype(), ids=str)
def test_affine_backward(factory_kwargs):
    factory_kwargs |= {"requires_grad": True}
    w = torch.rand(2, 4, 5, **factory_kwargs)
    b = torch.rand(2, 1, 5, **factory_kwargs)
    w_, b_ = [t for t in utils.detach_clone_req_grad(w, b)]

    layer = Linear(2, 4, 5)
    layer.weight = nn.Parameter(w)
    layer.bias = nn.Parameter(b)

    for x, dy in yield_x_and_dy(factory_kwargs):
        x_, dy_ = [t for t in utils.detach_clone_req_grad(x, dy)]

        layer(x).backward(dy)
        loop_linear(x_, w_, b_).backward(dy_)

        utils.assert_allclose(x.grad, x_.grad)
        utils.assert_allclose(layer.weight.grad, w_.grad)
        utils.assert_allclose(layer.bias.grad, b_.grad)


@pytest.mark.parametrize("factory_kwargs", utils.get_device_dtype(), ids=str)
def test_linear_backward(factory_kwargs):
    factory_kwargs |= {"requires_grad": True}
    w = torch.rand(2, 4, 5, **factory_kwargs)
    w_ = next(utils.detach_clone_req_grad(w))

    layer = Linear(2, 4, 5, bias=False)
    layer.weight = nn.Parameter(w)

    for x, dy in yield_x_and_dy(factory_kwargs):
        x_, dy_ = [t for t in utils.detach_clone_req_grad(x, dy)]

        layer(x).backward(dy)
        loop_linear(x_, w_, [None, None]).backward(dy_)

        utils.assert_allclose(x.grad, x_.grad)
        utils.assert_allclose(layer.weight.grad, w_.grad)
