import pytest
import torch

from rnam import minGRU
from tests import utils


# slightly relax tolerances for result validation
def assert_allclose(a, b):
    utils.assert_allclose(a, b, rtol=1e-4, atol=1e-7)


# allclose assertions assume that the sequential branch is implemented correctly
# (chunked) scan results are validated against this reference
def sequential(layer, x):
    all_hidden, prev_hidden = [], None
    for t in range(x.size(-2)):
        x_t = x[:, :, t : t + 1]
        hidden, prev_hidden = layer(x_t, prev_hidden)
        all_hidden.append(hidden)
    return torch.cat(all_hidden, dim=-2)


def scan(layer, x):
    return layer(x)[0]


def chunked_scan(layer, x):
    all_hidden, prev_hidden = [], None
    for t in range(x.size(-2) // 2):
        chunk = x[:, :, t * 2 : (t + 1) * 2]
        hidden, prev_hidden = layer(chunk, prev_hidden)
        all_hidden.append(hidden)
    return torch.cat(all_hidden, dim=-2)


@pytest.mark.parametrize("factory_kwargs", utils.get_device_dtype(), ids=str)
def test_forward(factory_kwargs):
    x = torch.rand(2, 3, 4, 5, **factory_kwargs)

    device = factory_kwargs["device"]
    layers = [minGRU(2, 5, exp).to(device) for exp in [1.0, 2.0, 0.5]]

    for layer in layers:
        y_seq = sequential(layer, x)
        y_scan = scan(layer, x)
        y_chunked = chunked_scan(layer, x)

        assert_allclose(y_scan, y_seq)
        assert_allclose(y_chunked, y_seq)


def detach_clone_req_grad(x):
    return next(utils.detach_clone_req_grad(x))


@pytest.mark.parametrize("factory_kwargs", utils.get_device_dtype(), ids=str)
def test_backward(factory_kwargs):
    x = torch.rand(2, 3, 4, 5, **factory_kwargs, requires_grad=True)
    dy = torch.rand_like(x)

    device = factory_kwargs["device"]
    layer = minGRU(2, 5, 2.0).to(device)

    grads = {}
    for fn in [sequential, scan, chunked_scan]:
        x_ = detach_clone_req_grad(x)
        y = fn(layer, x_)
        y.backward(dy)

        grads[fn.__name__] = {
            "dx": detach_clone_req_grad(x_.grad),
            "dw1": detach_clone_req_grad(layer.to_hidden_and_gate.weight.grad),
            "dw2": detach_clone_req_grad(layer.to_out.weight.grad),
        }

        layer.zero_grad()

    for fn_key in grads.keys():
        if fn_key == "sequential":
            continue

        for grad_key in grads[fn_key].keys():
            assert_allclose(
                grads[fn_key][grad_key], grads["sequential"][grad_key]
            )


@pytest.mark.parametrize("factory_kwargs", utils.get_device_dtype(), ids=str)
def test_perf(factory_kwargs):
    cuda = factory_kwargs["device"] == "cuda"
    x = torch.rand(2, 3, 1_000, 5, **factory_kwargs)

    device = factory_kwargs["device"]
    layer = minGRU(2, 5, 0.5).to(device)

    with utils.timer(cuda) as t_seq:
        _, _ = sequential(layer, x)

    with utils.timer(cuda) as t_scan:
        _, _ = scan(layer, x)

    # scan should be significantly faster than sequential
    assert t_scan.time < t_seq.time / 5, f"{t_seq.time:.6f}, {t_scan.time:.6f}"
