"""Voice input for hearthagent-pro. Records via parecord (PulseAudio's
recorder -- the one that actually works through WSL2's audio bridge,
unlike sounddevice's blocking API), transcribes locally with Whisper,
sends the text through the normal agent session.
Run with: uv run python3 voice_main.py
"""
import subprocess
import time

import numpy as np
from faster_whisper import WhisperModel

from agent.graph import Session

SAMPLE_RATE = 16000
DEFAULT_DURATION = 6  # seconds to record per prompt
RECORDING_PATH = "/tmp/hearthagent_voice_input.wav"

_whisper_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        print("Loading local Whisper model (first time only, this can take a moment)...")
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


def record_audio(duration: int = DEFAULT_DURATION) -> str:
    print(f"Recording for {duration}s... speak now.")
    proc = subprocess.Popen(
        ["parecord", "--channels=1", f"--rate={SAMPLE_RATE}", RECORDING_PATH]
    )
    time.sleep(duration)
    proc.terminate()
    proc.wait()
    print("Done recording.")
    return RECORDING_PATH


def transcribe(wav_path: str) -> str:
    model = _get_whisper()
    segments, _ = model.transcribe(wav_path, language="en")
    text = " ".join(seg.text.strip() for seg in segments)
    return text.strip()


def voice_session():
    print("hearthagent-pro -- voice input mode")
    print("Press Enter to start a 6-second recording, or type 'text' to switch to typed input, or '/quit' to exit.\n")
    session = Session()

    while True:
        choice = input("Press Enter to speak (or type a command)> ").strip()

        if choice == "/quit":
            print("Exiting.")
            break

        if choice.lower() == "text":
            typed = input("you> ").strip()
            if typed == "/quit":
                print("Exiting.")
                break
            if typed:
                session.send(typed)
            continue

        wav_path = record_audio()
        transcript = transcribe(wav_path)

        if not transcript:
            print("Heard nothing usable -- try again, speak clearly and not too fast.\n")
            continue

        print(f'Transcribed: "{transcript}"')
        confirm = input("Send this to the agent? [Y/n/edit] ").strip().lower()

        if confirm == "n":
            print("Discarded.\n")
            continue
        if confirm == "edit":
            transcript = input("Edit transcript> ").strip() or transcript

        session.send(transcript)


if __name__ == "__main__":
    voice_session()
