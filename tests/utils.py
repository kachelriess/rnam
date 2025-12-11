import torch


def get_device_dtype():
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    dtypes = [torch.float32]
    return [{"device": dev, "dtype": dt} for dev in devices for dt in dtypes]


def assert_allclose(a, b, rtol=1e-5, atol=1e-8):
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


def detach_clone_req_grad(*input):
    for x in input:
        x_ = x.detach().clone()
        x_.requires_grad = True
        yield x_
