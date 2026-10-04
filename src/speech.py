"""
speech.py
Speech-to-text for the Voice Complaints page.

Uses faster-whisper: OpenAI's Whisper speech-recognition model, run on the CPU.
- MODEL_SIZE "base.en" is Whisper's English-only base model (about 145 MB). It downloads
  automatically the first time it is used and is cached after that.
- Audio can be WAV, MP3, M4A, OGG, FLAC, WEBM or most phone voice-note formats. faster-whisper
  decodes it with PyAV, which bundles its own FFmpeg, so nothing else needs installing.
- Recordings longer than 30 seconds are handled in pieces, and long silences are skipped
  (vad_filter), which also stops Whisper inventing words during silence.
"""

MODEL_SIZE = "base.en"


def load_model(size=MODEL_SIZE):
    """Load the Whisper model. Imported here, not at the top of the file, so the rest of the
    app still runs on a machine where faster-whisper isn't installed."""
    from faster_whisper import WhisperModel
    return WhisperModel(size, device="cpu", compute_type="int8")


def transcribe(model, audio):
    """Turn speech into text. `audio` is a file path or a file-like object.
    Returns (text, duration_in_seconds)."""
    segments, info = model.transcribe(
        audio,
        language="en",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text, float(info.duration)
