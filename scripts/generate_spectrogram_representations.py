#!/usr/bin/env python3
"""
Generate comparable visualizations for the spectral representations discussed in the thesis.

The script reads one audio file, computes:
- spectrogram (linear STFT)
- mel spectrogram
- linear triangular projection
- adaptive combined log-linear projection

It saves one composite figure plus one image per representation.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

# Keep matplotlib cache writable even in restricted environments.
cache_root = Path(tempfile.gettempdir()) / "codex-plot-cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_audio = repo_root / "backups" / "bird_classification_edge" / "bird_sound_dataset" / "Apus_apus" / "CommonSwift.mp3"
    default_out = repo_root / "thesis" / "images" / "generated" / "commonswift_representations"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=default_audio, help="Input audio file.")
    parser.add_argument("--output-dir", type=Path, default=default_out, help="Directory for generated PNGs.")
    parser.add_argument("--sample-rate", type=int, default=32000, help="Resampling rate.")
    parser.add_argument("--n-fft", type=int, default=1024, help="FFT size.")
    parser.add_argument("--hop-length", type=int, default=320, help="STFT hop length.")
    parser.add_argument("--f-min", type=float, default=150.0, help="Minimum analysis frequency in Hz.")
    parser.add_argument("--f-max", type=float, default=16000.0, help="Maximum analysis frequency in Hz.")
    parser.add_argument("--n-filters", type=int, default=64, help="Number of projected filters.")
    parser.add_argument("--breakpoint", type=float, default=4000.0, help="Adaptive log-linear breakpoint in Hz.")
    parser.add_argument("--transition-width", type=float, default=100.0, help="Adaptive transition width.")
    parser.add_argument("--top-db", type=float, default=80.0, help="Dynamic range clipping in dB.")
    return parser.parse_args()


def load_audio_ffmpeg(path: Path, sample_rate: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError(f"Decoded audio is empty: {path}")
    return audio


def frame_signal(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if audio.size < frame_length:
        audio = np.pad(audio, (0, frame_length - audio.size))

    num_frames = 1 + int(np.ceil((audio.size - frame_length) / hop_length))
    padded_len = frame_length + hop_length * max(0, num_frames - 1)
    padded = np.pad(audio, (0, max(0, padded_len - audio.size)))

    shape = (num_frames, frame_length)
    strides = (padded.strides[0] * hop_length, padded.strides[0])
    return np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides, writeable=False)


def stft_magnitude(audio: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    frames = frame_signal(audio, n_fft, hop_length)
    window = np.hanning(n_fft).astype(np.float32)
    windowed = frames * window[None, :]
    spectrum = np.fft.rfft(windowed, n=n_fft, axis=1)
    magnitude = np.abs(spectrum).T
    return magnitude.astype(np.float32)


def hz_to_mel(freq_hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq_hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def hz_to_fft_bin(freq_hz: np.ndarray, sample_rate: int, n_fft: int) -> np.ndarray:
    return np.clip(np.round(freq_hz / (sample_rate / 2.0) * (n_fft // 2)).astype(int), 1, (n_fft // 2) - 2)


def generate_triangular_filterbank_from_centers(center_bins: np.ndarray, n_fft: int) -> np.ndarray:
    n_filters = len(center_bins)
    n_freq_bins = n_fft // 2 + 1
    filters = np.zeros((n_filters, n_freq_bins), dtype=np.float32)

    for i in range(n_filters):
        left = center_bins[i - 1] if i > 0 else 0
        center = center_bins[i]
        right = center_bins[i + 1] if i < n_filters - 1 else n_freq_bins - 1

        if center > left:
            filters[i, left:center + 1] = np.linspace(0.0, 1.0, center - left + 1, dtype=np.float32)
        if right > center:
            filters[i, center:right + 1] = np.maximum(
                filters[i, center:right + 1],
                np.linspace(1.0, 0.0, right - center + 1, dtype=np.float32),
            )

    return filters


def create_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_filters: int,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    mels = np.linspace(hz_to_mel(np.array([f_min]))[0], hz_to_mel(np.array([f_max]))[0], n_filters)
    center_hz = mel_to_hz(mels)
    center_bins = hz_to_fft_bin(center_hz, sample_rate, n_fft)
    return generate_triangular_filterbank_from_centers(center_bins, n_fft)


def create_linear_triangular_filterbank(
    sample_rate: int,
    n_fft: int,
    n_filters: int,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    center_hz = np.linspace(f_min, f_max, n_filters)
    center_bins = hz_to_fft_bin(center_hz, sample_rate, n_fft)
    return generate_triangular_filterbank_from_centers(center_bins, n_fft)


def create_adaptive_log_linear_filterbank(
    sample_rate: int,
    n_fft: int,
    n_filters: int,
    f_min: float,
    f_max: float,
    breakpoint: float,
    transition_width: float,
) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n_filters, dtype=np.float32)
    log_part = f_min * (f_max / f_min) ** x
    linear_part = f_min + x * (f_max - f_min)
    normalized_breakpoint = (breakpoint - f_min) / (f_max - f_min)
    gate = 1.0 / (1.0 + np.exp(-(x - normalized_breakpoint) * transition_width))
    center_hz = (1.0 - gate) * log_part + gate * linear_part
    center_bins = hz_to_fft_bin(center_hz, sample_rate, n_fft)
    return generate_triangular_filterbank_from_centers(center_bins, n_fft)


def apply_filterbank(magnitude: np.ndarray, filterbank: np.ndarray) -> np.ndarray:
    return filterbank @ magnitude


def to_db(spec: np.ndarray, top_db: float) -> np.ndarray:
    spec = np.maximum(spec, 1e-10)
    db = 20.0 * np.log10(spec / spec.max())
    return np.clip(db, -top_db, 0.0)


def plot_single_representation(
    data_db: np.ndarray,
    out_path: Path,
    title: str,
    sample_rate: int,
    hop_length: int,
    max_freq_hz: float,
) -> None:
    n_frames = data_db.shape[1]
    duration_s = n_frames * hop_length / sample_rate
    extent = [0.0, duration_s, 0.0, max_freq_hz / 1000.0]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    im = ax.imshow(data_db, origin="lower", aspect="auto", extent=extent, cmap="magma", vmin=data_db.min(), vmax=0.0)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Relative level (dB)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_composite(
    waveform: np.ndarray,
    representations: list[tuple[str, np.ndarray, float]],
    out_path: Path,
    sample_rate: int,
    hop_length: int,
) -> None:
    fig = plt.figure(figsize=(13, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.85, 1.0, 1.0])

    time_axis = np.arange(waveform.size, dtype=np.float32) / sample_rate
    waveform_ax = fig.add_subplot(gs[0, :])
    waveform_ax.plot(time_axis, waveform, color="#1f77b4", linewidth=0.8)
    waveform_ax.set_title("Waveform")
    waveform_ax.set_xlabel("Time (s)")
    waveform_ax.set_ylabel("Amplitude")
    waveform_ax.set_xlim(0.0, time_axis[-1] if time_axis.size else 0.0)

    rep_axes = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
    ]

    for ax, (title, data_db, max_freq_hz) in zip(rep_axes, representations):
        n_frames = data_db.shape[1]
        duration_s = n_frames * hop_length / sample_rate
        extent = [0.0, duration_s, 0.0, max_freq_hz / 1000.0]
        im = ax.imshow(
            data_db,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="magma",
            vmin=data_db.min(),
            vmax=0.0,
        )
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (kHz)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Spectral Representations for CommonSwift.mp3", fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audio = load_audio_ffmpeg(args.audio, args.sample_rate)
    magnitude = stft_magnitude(audio, args.n_fft, args.hop_length)

    mel_fb = create_mel_filterbank(args.sample_rate, args.n_fft, args.n_filters, args.f_min, args.f_max)
    linear_fb = create_linear_triangular_filterbank(args.sample_rate, args.n_fft, args.n_filters, args.f_min, args.f_max)
    adaptive_fb = create_adaptive_log_linear_filterbank(
        args.sample_rate,
        args.n_fft,
        args.n_filters,
        args.f_min,
        args.f_max,
        args.breakpoint,
        args.transition_width,
    )

    nyquist = args.sample_rate / 2.0
    representations = [
        ("Spectrogram (Linear STFT)", to_db(magnitude, args.top_db), nyquist),
        ("Mel Spectrogram", to_db(apply_filterbank(magnitude, mel_fb), args.top_db), args.f_max),
        ("Linear Triangular Projection", to_db(apply_filterbank(magnitude, linear_fb), args.top_db), args.f_max),
        (
            "Adaptive Combined Log-Linear",
            to_db(apply_filterbank(magnitude, adaptive_fb), args.top_db),
            args.f_max,
        ),
    ]

    safe_stem = args.audio.stem.replace(" ", "_").lower()
    plot_composite(
        audio,
        representations,
        args.output_dir / f"{safe_stem}_representations_grid.png",
        args.sample_rate,
        args.hop_length,
    )

    for title, data_db, max_freq_hz in representations:
        file_name = title.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "") + ".png"
        plot_single_representation(
            data_db,
            args.output_dir / file_name,
            title,
            args.sample_rate,
            args.hop_length,
            max_freq_hz,
        )

    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
