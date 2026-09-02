import torch
import torchaudio
from typing import List
import time

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add_noise_ratio(waveform: torch.Tensor, noise_ratio: float = 0.1) -> torch.Tensor:
    """
    Adds a specific ratio of noise to an audio signal.

    Args:
        waveform: The input audio waveform as a PyTorch tensor.
        noise_ratio: The ratio of noise to add to the signal (0.0 to 1.0).

    Returns:
        The noisy audio waveform as a PyTorch tensor.
    """

    noise = torch.randn_like(waveform)
    noisy_waveform = waveform + noise_ratio * noise
    return noisy_waveform

def noise(waveform: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(waveform)
    assert noise.shape == waveform.shape, "The generated noise has differet size"
    return noise


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels: List, blank: int = 0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])
    
if __name__ == "__main__":
    # Model
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    
    # Data
    dataset = torchaudio.datasets.YESNO(
        root="./audio",
        download=True
    )

    sample_index = 0
    waveform, sample_rate, label = dataset[sample_index]

    # Preprocess the dataset with specific sample rate
    waveform, sample_rate, label = dataset[1]
    waveform = waveform.to(device)

    processing_start_time = time.time()

    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
    
    with torch.inference_mode():
        emission, _ = model(waveform)
    
    # Decode step
    decoder = GreedyCTCDecoder(labels=bundle.get_labels())
    transcript = decoder(emission[0])

    print(f"Transcript: {transcript}")

    processing_end_time = time.time()
    print(f"The total processing time: {processing_end_time - processing_start_time}")