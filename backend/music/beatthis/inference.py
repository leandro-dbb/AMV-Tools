# Vendored from beat_this 1.1.0 (MIT, CPJKU), see LICENSE-beat-this:
# load_model, split_piece, aggregate_prediction, split_predict_aggregate are
# unchanged; the checkpoint is a local file (downloaded by the model
# registry), and the log-mel front end is rebuilt on torch.stft so that
# torchaudio (whose wheels must match each torch build) is not needed.
import inspect
import math

import numpy as np
import torch
import torch.nn.functional as F

from .beat_tracker import BeatThis
from .utils import replace_state_dict_key


def load_checkpoint(checkpoint_path: str, device="cpu") -> dict:
    return torch.load(checkpoint_path, map_location=device, weights_only=True)


def load_model(
    checkpoint_path: str | None = "final0", device: str | torch.device = "cpu"
) -> BeatThis:
    """
    Load a BeatThis model from a checkpoint.

    Args:
        checkpoint_path (str, optional): The path to the checkpoint. Can be a local path, a URL, or a shortname.
        device (torch.device or str): The device to load the model on.

    Returns:
        BeatThis: The loaded model.
    """
    if checkpoint_path is not None:
        checkpoint = load_checkpoint(checkpoint_path, device)
        # Retrieve the model hyperparameters as it could be the small model
        hparams = checkpoint["hyper_parameters"]
        # Filter only those hyperparameters that apply to the model itself
        hparams = {
            k: v
            for k, v in hparams.items()
            if k in set(inspect.signature(BeatThis).parameters)
        }
        # Create the uninitialized model
        model = BeatThis(**hparams)
        # The PLBeatThis (LightningModule) state_dict contains the BeatThis
        # state_dict under the "model." prefix; remove the prefix to load it
        state_dict = replace_state_dict_key(checkpoint["state_dict"], "model.", "")
        model.load_state_dict(state_dict)
    else:
        model = BeatThis()
    return model.to(device).eval()


def zeropad(spect: torch.Tensor, left: int = 0, right: int = 0):
    """
    Pads a tensor spectrogram matrix of shape (time x bins) with `left` frames in the beginning and `right` frames in the end.
    """
    if left == 0 and right == 0:
        return spect
    else:
        return F.pad(spect, (0, 0, left, right), "constant", 0)


def split_piece(
    spect: torch.Tensor,
    chunk_size: int,
    border_size: int = 6,
    avoid_short_end: bool = True,
):
    """
    Split a tensor spectrogram matrix of shape (time x bins) into time chunks of `chunk_size` and return the chunks and starting positions.
    The `border_size` is the number of frames assumed to be discarded in the predictions on either side (since the model was not trained on the input edges due to the max-pool in the loss).
    To cater for this, the first and last chunk are padded by `border_size` on the beginning and end, respectively, and consecutive chunks overlap by `border_size`.
    If `avoid_short_end` is true, the last chunk start is shifted left to ends at the end of the piece, therefore the last chunk can potentially overlap with previous chunks more than border_size, otherwise it will be a shorter segment.
    If the piece is shorter than `chunk_size`, avoid_short_end is ignored and the piece is returned as a single shorter chunk.

    Args:
        spect (torch.Tensor): The input spectrogram tensor of shape (time x bins).
        chunk_size (int): The size of the chunks to produce.
        border_size (int, optional): The size of the border to overlap between chunks. Defaults to 6.
        avoid_short_end (bool, optional): If True, the last chunk is shifted left to end at the end of the piece. Defaults to True.
    """
    # generate the start and end indices
    starts = np.arange(
        -border_size, len(spect) - border_size, chunk_size - 2 * border_size
    )
    if avoid_short_end and len(spect) > chunk_size - 2 * border_size:
        # if we avoid short ends, move the last index to the end of the piece - (chunk_size - border_size)
        starts[-1] = len(spect) - (chunk_size - border_size)
    # generate the chunks
    chunks = [
        zeropad(
            spect[max(start, 0) : min(start + chunk_size, len(spect))],
            left=max(0, -start),
            right=max(0, min(border_size, start + chunk_size - len(spect))),
        )
        for start in starts
    ]
    return chunks, starts


def aggregate_prediction(
    pred_chunks: list,
    starts: list,
    full_size: int,
    chunk_size: int,
    border_size: int,
    overlap_mode: str,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregates the predictions for the whole piece based on the given prediction chunks.

    Args:
        pred_chunks (list): List of prediction chunks, where each chunk is a dictionary containing 'beat' and 'downbeat' predictions.
        starts (list): List of start positions for each prediction chunk.
        full_size (int): Size of the full piece.
        chunk_size (int): Size of each prediction chunk.
        border_size (int): Size of the border to be discarded from each prediction chunk.
        overlap_mode (str): Mode for handling overlapping predictions. Can be 'keep_first' or 'keep_last'.
        device (torch.device): Device to be used for the predictions.

    Returns:
        tuple: A tuple containing the aggregated beat predictions and downbeat predictions as torch tensors for the whole piece.
    """
    if border_size > 0:
        # cut the predictions to discard the border
        pred_chunks = [
            {
                "beat": pchunk["beat"][border_size:-border_size],
                "downbeat": pchunk["downbeat"][border_size:-border_size],
            }
            for pchunk in pred_chunks
        ]
    # aggregate the predictions for the whole piece
    piece_prediction_beat = torch.full((full_size,), -1000.0, device=device)
    piece_prediction_downbeat = torch.full((full_size,), -1000.0, device=device)
    if overlap_mode == "keep_first":
        # process in reverse order, so predictions of earlier excerpts overwrite later ones
        pred_chunks = reversed(list(pred_chunks))
        starts = reversed(list(starts))
    for start, pchunk in zip(starts, pred_chunks):
        piece_prediction_beat[
            start + border_size : start + chunk_size - border_size
        ] = pchunk["beat"]
        piece_prediction_downbeat[
            start + border_size : start + chunk_size - border_size
        ] = pchunk["downbeat"]
    return piece_prediction_beat, piece_prediction_downbeat


def split_predict_aggregate(
    spect: torch.Tensor,
    chunk_size: int,
    border_size: int,
    overlap_mode: str,
    model: torch.nn.Module,
) -> dict:
    """
    Function for pieces that are longer than the training length of the model.
    Split the input piece into chunks, run the model on them, and aggregate the predictions.
    The spect is supposed to be a torch tensor of shape (time x bins), i.e., unbatched, and the output is also unbatched.

    Args:
        spect (torch.Tensor): the input piece
        chunk_size (int): the length of the chunks
        border_size (int): the size of the border that is discarded from the predictions
        overlap_mode (str): how to handle overlaps between chunks
        model (torch.nn.Module): the model to run

    Returns:
        dict: the model framewise predictions for the hole piece as a dictionary containing 'beat' and 'downbeat' predictions.
    """
    # split the piece into chunks
    chunks, starts = split_piece(
        spect, chunk_size, border_size=border_size, avoid_short_end=True
    )
    # run the model
    pred_chunks = [model(chunk.unsqueeze(0)) for chunk in chunks]
    # remove the extra dimension in beat and downbeat prediction due to batch size 1
    pred_chunks = [
        {"beat": p["beat"][0], "downbeat": p["downbeat"][0]} for p in pred_chunks
    ]
    piece_prediction_beat, piece_prediction_downbeat = aggregate_prediction(
        pred_chunks,
        starts,
        spect.shape[0],
        chunk_size,
        border_size,
        overlap_mode,
        spect.device,
    )
    # save it to model_prediction
    return {"beat": piece_prediction_beat, "downbeat": piece_prediction_downbeat}




def _hz_to_mel_slaney(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, mels)


def _mel_to_hz_slaney(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)


def mel_filterbank(sample_rate=22050, n_fft=1024, f_min=30.0, f_max=11000.0, n_mels=128) -> np.ndarray:
    """torchaudio.functional.melscale_fbanks(norm=None, mel_scale="slaney"):
    (n_fft // 2 + 1, n_mels) triangular filters."""
    all_freqs = np.linspace(0, sample_rate // 2, n_fft // 2 + 1)
    m_pts = np.linspace(_hz_to_mel_slaney(f_min), _hz_to_mel_slaney(f_max), n_mels + 2)
    f_pts = _mel_to_hz_slaney(m_pts)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts[None, :] - all_freqs[:, None]
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    return np.maximum(0.0, np.minimum(down, up)).astype(np.float32)


class LogMelSpect(torch.nn.Module):
    """beat_this.preprocessing.LogMelSpect (torchaudio MelSpectrogram with
    n_fft 1024, hop 441, 30-11000 Hz, 128 slaney mels, magnitude, frame-length
    normalisation; then log1p(1000 x)), on torch.stft."""

    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=441, f_min=30, f_max=11000,
                 n_mels=128, log_multiplier=1000, device="cpu"):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop_length
        self.window = torch.hann_window(n_fft, device=device)
        self.fb = torch.tensor(mel_filterbank(sample_rate, n_fft, f_min, f_max, n_mels), device=device)
        self.log_multiplier = log_multiplier

    def forward(self, x):
        spec = torch.stft(x, self.n_fft, hop_length=self.hop, win_length=self.n_fft, window=self.window,
                          center=True, pad_mode="reflect", normalized=True, onesided=True,
                          return_complex=True).abs()   # "frame_length": stft's own 1/sqrt(n_fft)
        mel = (spec.T @ self.fb)
        return torch.log1p(self.log_multiplier * mel)


class Audio2Beats:
    """Mono float signal at ``sr`` -> (beats, downbeats) in seconds."""

    def __init__(self, checkpoint_path: str, device="cpu"):
        self.device = torch.device(device)
        self.model = load_model(checkpoint_path, self.device)
        self.spect = LogMelSpect(device=self.device)
        from .postprocessor import Postprocessor
        self.frames2beats = Postprocessor(type="minimal")

    def __call__(self, signal: np.ndarray, sr: int):
        if sr != 22050:
            import soxr
            signal = soxr.resample(signal, in_rate=sr, out_rate=22050)
        x = torch.tensor(np.asarray(signal, dtype=np.float32), device=self.device)
        with torch.inference_mode():
            spect = self.spect(x)
            pred = split_predict_aggregate(spect=spect, chunk_size=1500, overlap_mode="keep_first",
                                           border_size=6, model=self.model)
        return self.frames2beats(pred["beat"].float(), pred["downbeat"].float())
