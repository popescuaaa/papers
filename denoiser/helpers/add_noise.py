import torch
from typing import Optional

def add_noise(
    waveform: torch.Tensor, 
    noise: torch.Tensor, 
    snr: torch.Tensor, 
    lengths: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""Scales and adds noise to waveform per signal-to-noise ratio.

    Specifically, for each pair of waveform vector :math:`x \in \mathbb{R}^L` and noise vector
    :math:`n \in \mathbb{R}^L`, the function computes output :math:`y` as

    .. math::
        y = x + a n \, \text{,}

    where

    .. math::
        a = \sqrt{ \frac{ ||x||_{2}^{2} }{ ||n||_{2}^{2} } \cdot 10^{-\frac{\text{SNR}}{10}} } \, \text{,}

    with :math:`\text{SNR}` being the desired signal-to-noise ratio between :math:`x` and :math:`n`, in dB.

    Note that this function broadcasts singleton leading dimensions in its inputs in a manner that is
    consistent with the above formulae and PyTorch's broadcasting semantics.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (torch.Tensor): Input waveform, with shape `(..., L)`.
        noise (torch.Tensor): Noise, with shape `(..., L)` (same shape as ``waveform``).
        snr (torch.Tensor): Signal-to-noise ratios in dB, with shape `(...,)`.
        lengths (torch.Tensor or None, optional): Valid lengths of signals in ``waveform`` and ``noise``, with shape
            `(...,)` (leading dimensions must match those of ``waveform``). If ``None``, all elements in ``waveform``
            and ``noise`` are treated as valid. (Default: ``None``)

    Returns:
        torch.Tensor: Result of scaling and adding ``noise`` to ``waveform``, with shape `(..., L)`
        (same shape as ``waveform``).
    """
    
    if not (waveform.ndim - 1 == noise.ndim - 1 == snr.ndim and (lengths is None or lengths.ndim == snr.ndim)):
        raise ValueError("Input leading dimensions don't match.")

    L = waveform.size(-1)

    if L != noise.size(-1):
        raise ValueError(f"Length dimensions of waveform and noise don't match (got {L} and {noise.size(-1)}).")

    # compute scale
    if lengths is not None:
        print("Lengths is not specified")
        mask = torch.arange(0, L, device=lengths.device).expand(waveform.shape) < lengths.unsqueeze(
            -1
        )  # (*, L) < (*, 1) = (*, L)
        masked_waveform = waveform * mask
        masked_noise = noise * mask
    else:
        masked_waveform = waveform
        masked_noise = noise

    energy_signal = torch.linalg.vector_norm(masked_waveform, ord=2, dim=-1) ** 2  # (*,)
    print("energy signal computed")
    energy_noise = torch.linalg.vector_norm(masked_noise, ord=2, dim=-1) ** 2  # (*,)
    print("energy noise computed")
    original_snr_db = 10 * (torch.log2(energy_signal) - torch.log2(energy_noise))
    
    print("snr computed")
    scale = 10 ** ((original_snr_db - snr) / 20.0)  # (*,)
    print("scale computed")
    
    # scale noise
    scaled_noise = scale.unsqueeze(-1) * noise  # (*, 1) * (*, L) = (*, L)

    return waveform + scaled_noise  # (*, L)


