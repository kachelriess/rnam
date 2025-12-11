import pytest
import torch


@pytest.fixture(autouse=True, scope="session")
def set_reproducibility():
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # init cuda context
        x = torch.rand(1, device="cuda", requires_grad=True)
        x.backward(torch.ones_like(x))
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.fp32_precision = "high"
