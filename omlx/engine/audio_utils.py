# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for audio engines (STT, TTS, STS)."""

import io
import shutil
import subprocess
import wave

import numpy as np


# Default sample rate used when the model does not report one.
DEFAULT_SAMPLE_RATE = 24000

# OpenAI-compatible response_format values for /v1/audio/speech.
# Maps each value to its HTTP Content-Type header.
TTS_FORMAT_MEDIA_TYPES: dict[str, str] = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}

SUPPORTED_TTS_FORMATS: frozenset[str] = frozenset(TTS_FORMAT_MEDIA_TYPES.keys())


def audio_to_wav_bytes(audio_array, sample_rate: int) -> bytes:
    """Convert a float32 audio array to 16-bit PCM WAV bytes.

    Args:
        audio_array: numpy or mlx array of float32 samples in [-1, 1]
        sample_rate: audio sample rate in Hz

    Returns:
        WAV-encoded bytes (RIFF header + PCM data)
    """
    # Ensure we have a numpy array for the wave module
    if not isinstance(audio_array, np.ndarray):
        # NumPy doesn't support bfloat16 — cast to float32 first
        if hasattr(audio_array, "dtype"):
            import mlx.core as mx

            if audio_array.dtype == mx.bfloat16:
                audio_array = audio_array.astype(mx.float32)
        audio_array = np.array(audio_array)

    # Flatten to 1-D (mono)
    audio_array = audio_array.flatten()

    # Clip to [-1, 1] then convert to int16
    audio_array = np.clip(audio_array, -1.0, 1.0)
    audio_int16 = (audio_array * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Transcoding (OpenAI response_format support — see #753)
# ---------------------------------------------------------------------------


class TranscodeError(RuntimeError):
    """Raised when audio format conversion fails at runtime."""


def _extract_pcm_from_wav(wav_bytes: bytes) -> bytes:
    """Return raw 16-bit little-endian PCM samples from WAV container bytes."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.readframes(wf.getnframes())


def _ffmpeg_transcode(wav_bytes: bytes, args: list[str]) -> bytes:
    """Pipe WAV bytes through ffmpeg with the given output args; return stdout."""
    if shutil.which("ffmpeg") is None:
        raise TranscodeError(
            "ffmpeg is required to transcode audio but was not found on PATH. "
            "Install ffmpeg (e.g. `brew install ffmpeg`) or request "
            "response_format='wav'."
        )

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", *args, "pipe:1"]
    try:
        proc = subprocess.run(
            cmd,
            input=wav_bytes,
            capture_output=True,
            check=False,
        )
    except OSError as exc:  # e.g. permission denied
        raise TranscodeError(f"Failed to invoke ffmpeg: {exc}") from exc

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise TranscodeError(f"ffmpeg exited with status {proc.returncode}: {stderr}")
    return proc.stdout


def transcode_wav_bytes(wav_bytes: bytes, target_format: str) -> bytes:
    """Convert WAV bytes to the requested OpenAI-compatible response_format.

    Args:
        wav_bytes: RIFF/WAV encoded audio from the TTS engine.
        target_format: One of ``SUPPORTED_TTS_FORMATS`` (case-insensitive).

    Returns:
        Encoded bytes in the requested format. For ``wav`` the input is
        returned unchanged. For ``pcm`` the RIFF header is stripped and the
        raw 16-bit LE sample data is returned. All other formats use ffmpeg.

    Raises:
        ValueError: if ``target_format`` is not supported.
        TranscodeError: if ffmpeg is missing or returns a non-zero status.
    """
    fmt = target_format.lower()
    if fmt not in SUPPORTED_TTS_FORMATS:
        raise ValueError(
            f"Unsupported response_format '{target_format}'. "
            f"Supported: {sorted(SUPPORTED_TTS_FORMATS)}"
        )

    if fmt == "wav":
        return wav_bytes
    if fmt == "pcm":
        return _extract_pcm_from_wav(wav_bytes)
    if fmt == "mp3":
        return _ffmpeg_transcode(
            wav_bytes, ["-vn", "-f", "mp3", "-codec:a", "libmp3lame", "-q:a", "2"]
        )
    if fmt == "opus":
        return _ffmpeg_transcode(
            wav_bytes, ["-vn", "-f", "ogg", "-codec:a", "libopus", "-b:a", "64k"]
        )
    if fmt == "flac":
        return _ffmpeg_transcode(wav_bytes, ["-vn", "-f", "flac", "-codec:a", "flac"])
    if fmt == "aac":
        return _ffmpeg_transcode(
            wav_bytes, ["-vn", "-f", "adts", "-codec:a", "aac", "-b:a", "128k"]
        )
    # Unreachable — guarded by SUPPORTED_TTS_FORMATS above.
    raise ValueError(f"Unhandled format: {fmt}")
