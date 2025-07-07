import torch

def config_pytorch(
    benchmark: bool = True,
    deterministic: bool = False
):
    """
    Initialise PyTorch/cuDNN global settings and return a `torch.device`
    that points to the first available GPU (or CPU if no GPU is found).

    Parameters
    ----------
    benchmark : bool, default **True**
        • If *True*  → on the *first* time cuDNN meets a new tensor shape
          (e.g. an image of size 256 × 256), it tries several algorithms,
          times them, then **remembers the fastest choice** for every later
          call with the same shape.  This gives a **speed-up** once the
          model “warms up”.  
        • If *False* → skip that search and pick a safe default.  Use this
          when your input size keeps changing every batch; otherwise the
          constant re-profiling can slow you down.

    deterministic : bool, default **False**
        • If *True*  → cuDNN is told to use **only algorithms that are
          mathematically deterministic**.  This makes your results
          *bit-for-bit reproducible* across runs and GPUs, but can lower
          throughput by 5-20 %.  
        • When *True* cuDNN automatically **disables `benchmark`** (the two
          flags are mutually exclusive).  
        • Leave it *False* for maximum speed when exact reproducibility is
          not critical (typical for inference).

    Returns
    -------
    torch.device
        *cuda* if a GPU is visible, otherwise *cpu*.

    Example
    -------
    >>> device = config_pytorch()               # fast, non-deterministic
    >>> device = config_pytorch(
    ...     benchmark=False, deterministic=True)  # slower, 100 % repeatable
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark     = benchmark
        torch.backends.cudnn.deterministic = deterministic
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device
