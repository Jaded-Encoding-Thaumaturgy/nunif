from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = SCRIPT_DIR / "waifu2x" / "onnx_models"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "waifu2x" / "onnx_models_flat"

SCALE_RE = re.compile(r"^scale\d+x$")
NOISE_RE = re.compile(r"^noise\d+$")


def build_flat_name(source_dir: Path, onnx_path: Path) -> str:
    relative_path = onnx_path.relative_to(source_dir)
    prefix_parts = list(relative_path.parent.parts)
    stem_parts = relative_path.stem.split("_")

    scales = [part for part in stem_parts if SCALE_RE.match(part)]
    noises = [part for part in stem_parts if NOISE_RE.match(part)]
    other_parts = [part for part in stem_parts if not SCALE_RE.match(part) and not NOISE_RE.match(part)]

    name_parts = prefix_parts + scales + noises + other_parts
    return "_".join(name_parts) + onnx_path.suffix


def iter_onnx_files(source_dir: Path):
    for onnx_path in sorted(source_dir.rglob("*.onnx")):
        if onnx_path.name == "scale1x.onnx":
            continue
        yield onnx_path


def copy_flattened(source_dir: Path, output_dir: Path, *, dry_run: bool = False, overwrite: bool = False) -> int:
    copied = 0
    planned_outputs: set[Path] = set()

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for onnx_path in iter_onnx_files(source_dir):
        output_path = output_dir / build_flat_name(source_dir, onnx_path)

        if output_path in planned_outputs:
            raise FileExistsError(f"Duplicate output name planned: {output_path.name}")
        planned_outputs.add(output_path)

        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

        print(f"{onnx_path.relative_to(source_dir)} -> {output_path.name}")
        if not dry_run:
            shutil.copy2(onnx_path, output_path)
        copied += 1

    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy waifu2x ONNX models into one folder with flattened names.")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Source model directory. Defaults to {DEFAULT_SOURCE_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Defaults to {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files that already exist in the output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned copies without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copied = copy_flattened(
        args.source.resolve(), args.output.resolve(), dry_run=args.dry_run, overwrite=args.overwrite
    )
    action = "Would copy" if args.dry_run else "Copied"
    print(f"{action} {copied} ONNX files.")


if __name__ == "__main__":
    main()
