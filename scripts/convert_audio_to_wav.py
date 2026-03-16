from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import soundfile as sf


def convert_to_wav(input_path: Path, output_path: Path) -> None:
    waveform, sample_rate = librosa.load(input_path, sr=None, mono=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, waveform, sample_rate, format="WAV")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an audio file such as MP3 to WAV/PCM for model preprocessing."
    )
    parser.add_argument("input_path", type=Path, help="Path to the source audio file.")
    parser.add_argument("output_path", type=Path, help="Path to the output WAV file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_to_wav(args.input_path, args.output_path)
    print(f"Converted '{args.input_path}' -> '{args.output_path}'")


if __name__ == "__main__":
    main()
