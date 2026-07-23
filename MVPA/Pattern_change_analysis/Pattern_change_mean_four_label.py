#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calculate stage-specific mean pattern-change images.

For each participant and each learning stage (LN, LE, RN, and RE), this
script averages all corresponding trial-wise pattern-change images while
preserving the original participant/stage directory structure.

Input structure
---------------
<input_root>/
├── Sub02/
│   ├── LN/
│   │   ├── pattern_change_trial_001.nii
│   │   ├── pattern_change_trial_002.nii
│   │   └── ...
│   ├── LE/
│   ├── RN/
│   └── RE/
├── Sub03/
│   └── ...
└── ...

Output structure
----------------
<output_root>/
├── Sub02/
│   ├── LN/
│   │   └── pattern_change_mean_LN.nii
│   ├── LE/
│   │   └── pattern_change_mean_LE.nii
│   ├── RN/
│   │   └── pattern_change_mean_RN.nii
│   └── RE/
│       └── pattern_change_mean_RE.nii
├── Sub03/
│   └── ...
├── stage_mean_summary.tsv
└── skipped_images.tsv

Public-release version
----------------------
- Contains no local drive letters, user names, or machine-specific paths.
- Input and output directories are supplied through command-line arguments.
- Relative default paths are used when arguments are omitted.
- Both .nii and .nii.gz inputs are supported.
- NaN and Inf values are excluded voxel by voxel.
- Images with mismatched dimensions or affine matrices are skipped.
- Output data are stored as float32.

Dependencies
------------
pip install numpy nibabel

Example
-------
python calculate_stage_mean_pattern_change.py \
    --input-root /path/to/pattern_change_labeled \
    --output-root /path/to/pattern_change_stage_means

Process selected participants only
----------------------------------
python calculate_stage_mean_pattern_change.py \
    --subject-ids 02 03 04 05

Generate compressed outputs
---------------------------
python calculate_stage_mean_pattern_change.py \
    --output-extension .nii.gz
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np


DEFAULT_LABELS = (
    "LN",
    "LE",
    "RN",
    "RE",
)

TRIAL_FILE_PATTERN = re.compile(
    r"^pattern_change_trial_(\d+)\.nii(?:\.gz)?$",
    flags=re.IGNORECASE,
)

SUBJECT_DIRECTORY_PATTERN = re.compile(
    r"^Sub(\d+)$",
    flags=re.IGNORECASE,
)


def parse_arguments() -> argparse.Namespace:
    """Parse portable input, output, and analysis settings."""
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Average trial-wise pattern-change images separately "
            "for LN, LE, RN, and RE."
        )
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=script_dir / "data" / "pattern_change_labeled",
        help=(
            "Root directory containing SubXX/LN, LE, RN, and RE folders. "
            "Default: <script_dir>/data/pattern_change_labeled"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=script_dir / "outputs" / "pattern_change_stage_means",
        help=(
            "Output directory for stage-specific mean images. "
            "Default: <script_dir>/outputs/pattern_change_stage_means"
        ),
    )

    parser.add_argument(
        "--subject-ids",
        nargs="+",
        default=None,
        help=(
            "Optional participant identifiers, for example 02 03 04. "
            "When omitted, all Sub* directories are detected automatically."
        ),
    )

    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(DEFAULT_LABELS),
        help=(
            "Stage labels to process. "
            "Default: LN LE RN RE"
        ),
    )

    parser.add_argument(
        "--output-extension",
        choices=(".nii", ".nii.gz"),
        default=".nii",
        help="Output image extension. Default: .nii",
    )

    parser.add_argument(
        "--affine-tolerance",
        type=float,
        default=1e-5,
        help=(
            "Absolute tolerance used when comparing affine matrices. "
            "Default: 1e-5"
        ),
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip a participant-stage combination when its output "
            "mean image already exists."
        ),
    )

    parser.add_argument(
        "--save-voxel-count",
        action="store_true",
        help=(
            "Also save a voxel-wise count image showing how many trial "
            "images contributed to each voxel."
        ),
    )

    return parser.parse_args()


def natural_sort_key(value: str) -> list[object]:
    """Sort strings naturally, for example Sub2 before Sub10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def normalize_subject_id(value: str) -> str:
    """
    Extract the numeric part of a participant identifier.

    Examples
    --------
    Sub02   -> 02
    sub-002 -> 002
    2       -> 2
    """
    match = re.search(
        r"(\d+)",
        str(value),
    )

    if match is None:
        raise ValueError(
            f"Could not parse a participant number from: {value}"
        )

    return match.group(1)


def normalize_labels(
    labels: Sequence[str],
) -> list[str]:
    """Normalize stage labels and remove duplicates."""
    normalized: list[str] = []

    for label in labels:
        clean_label = str(label).strip().upper()

        if not clean_label:
            continue

        if clean_label not in normalized:
            normalized.append(clean_label)

    if not normalized:
        raise ValueError(
            "At least one stage label is required."
        )

    return normalized


def detect_subject_directories(
    input_root: Path,
) -> list[Path]:
    """Detect participant directories matching Sub<number>."""
    subject_directories: list[Path] = []

    for path in input_root.iterdir():
        if not path.is_dir():
            continue

        if SUBJECT_DIRECTORY_PATTERN.fullmatch(path.name):
            subject_directories.append(path)

    return sorted(
        subject_directories,
        key=lambda path: natural_sort_key(path.name),
    )


def find_subject_directory(
    input_root: Path,
    subject_id: str,
) -> Path | None:
    """
    Locate a participant directory using common zero-padding forms.

    Supported examples
    ------------------
    Sub2
    Sub02
    Sub002
    """
    participant_number = int(subject_id)

    candidate_names = [
        f"Sub{subject_id}",
        f"Sub{participant_number}",
        f"Sub{participant_number:02d}",
        f"Sub{participant_number:03d}",
    ]

    checked_names: set[str] = set()

    for candidate_name in candidate_names:
        if candidate_name in checked_names:
            continue

        checked_names.add(candidate_name)

        candidate_path = (
            input_root
            / candidate_name
        )

        if candidate_path.is_dir():
            return candidate_path

    return None


def get_subject_directories(
    input_root: Path,
    requested_subject_ids: Sequence[str] | None,
) -> list[Path]:
    """Return requested participant directories or detect all participants."""
    if requested_subject_ids is None:
        return detect_subject_directories(
            input_root
        )

    subject_directories: list[Path] = []

    for requested_id in requested_subject_ids:
        subject_id = normalize_subject_id(
            requested_id
        )

        subject_directory = find_subject_directory(
            input_root,
            subject_id,
        )

        if subject_directory is None:
            print(
                f"[Missing] Participant directory was not found: "
                f"{requested_id}"
            )
            continue

        subject_directories.append(
            subject_directory
        )

    unique_directories = {
        path.resolve(): path
        for path in subject_directories
    }

    return sorted(
        unique_directories.values(),
        key=lambda path: natural_sort_key(path.name),
    )


def list_nifti_files(
    label_directory: Path,
) -> list[Path]:
    """
    Return trial-wise NIfTI files sorted by trial number.

    When both .nii and .nii.gz versions of the same trial exist, the
    compressed .nii.gz version is preferred to avoid counting the same
    trial twice.
    """
    files_by_trial: dict[int, Path] = {}

    if not label_directory.is_dir():
        return []

    for path in label_directory.iterdir():
        if not path.is_file():
            continue

        match = TRIAL_FILE_PATTERN.fullmatch(
            path.name
        )

        if match is None:
            continue

        trial_number = int(
            match.group(1)
        )

        existing_path = files_by_trial.get(
            trial_number
        )

        if existing_path is None:
            files_by_trial[trial_number] = path
            continue

        if (
            path.name.lower().endswith(".nii.gz")
            and not existing_path.name.lower().endswith(".nii.gz")
        ):
            files_by_trial[trial_number] = path

    return [
        files_by_trial[trial_number]
        for trial_number in sorted(files_by_trial)
    ]


def load_3d_image(
    path: Path,
) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    """
    Load a NIfTI image as a three-dimensional float32 array.

    Singleton four-dimensional images are reduced to three dimensions.
    """
    image = nib.load(
        str(path)
    )

    data = image.get_fdata(
        dtype=np.float32
    )

    if (
        data.ndim == 4
        and data.shape[-1] == 1
    ):
        data = data[..., 0]

    if data.ndim != 3:
        raise ValueError(
            "Expected a 3D image or a singleton 4D image, "
            f"but received shape {data.shape}"
        )

    return image, data


def grids_match(
    reference_shape: tuple[int, ...],
    reference_affine: np.ndarray,
    current_shape: tuple[int, ...],
    current_affine: np.ndarray,
    affine_tolerance: float,
) -> bool:
    """Check whether two images share dimensions and affine."""
    return (
        tuple(current_shape)
        == tuple(reference_shape)
        and np.allclose(
            current_affine,
            reference_affine,
            atol=affine_tolerance,
            rtol=0.0,
        )
    )


def compute_mean_nifti(
    nifti_paths: Sequence[Path],
    output_path: Path,
    *,
    affine_tolerance: float,
    save_voxel_count: bool,
) -> tuple[int, int, list[tuple[Path, str]]]:
    """
    Calculate and save a voxel-wise finite-value mean.

    Only images matching the first readable image's shape and affine are
    included. NaN and Inf values are excluded separately at each voxel.

    Returns
    -------
    number_used:
        Number of images included in the mean.
    number_found:
        Total number of candidate trial images.
    skipped_images:
        List of skipped image paths and reasons.
    """
    number_found = len(
        nifti_paths
    )

    if number_found == 0:
        return 0, 0, []

    template_image = None
    reference_shape = None
    reference_affine = None

    sum_array = None
    count_array = None

    number_used = 0
    skipped_images: list[tuple[Path, str]] = []

    for nifti_path in nifti_paths:
        try:
            image, data = load_3d_image(
                nifti_path
            )

        except Exception as error:
            reason = (
                f"read error: {error}"
            )

            skipped_images.append(
                (
                    nifti_path,
                    reason,
                )
            )

            print(
                f"[Skipped] Could not read {nifti_path}: {error}"
            )
            continue

        if template_image is None:
            template_image = image
            reference_shape = data.shape
            reference_affine = image.affine.copy()

            sum_array = np.zeros(
                reference_shape,
                dtype=np.float64,
            )

            count_array = np.zeros(
                reference_shape,
                dtype=np.uint32,
            )

        elif not grids_match(
            reference_shape=reference_shape,
            reference_affine=reference_affine,
            current_shape=data.shape,
            current_affine=image.affine,
            affine_tolerance=affine_tolerance,
        ):
            reason = (
                "image dimensions or affine do not match "
                "the reference grid"
            )

            skipped_images.append(
                (
                    nifti_path,
                    reason,
                )
            )

            print(
                f"[Skipped] Grid mismatch: {nifti_path}"
            )
            continue

        valid_values = np.isfinite(
            data
        )

        sum_array[valid_values] += (
            data[valid_values]
        )

        count_array[valid_values] += 1

        number_used += 1

    if (
        template_image is None
        or sum_array is None
        or count_array is None
        or number_used == 0
    ):
        return (
            0,
            number_found,
            skipped_images,
        )

    mean_volume = np.full(
        sum_array.shape,
        np.nan,
        dtype=np.float32,
    )

    np.divide(
        sum_array,
        count_array,
        out=mean_volume,
        where=count_array > 0,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_header = template_image.header.copy()
    output_header.set_data_dtype(
        np.float32
    )

    output_image = nib.Nifti1Image(
        mean_volume,
        affine=reference_affine,
        header=output_header,
    )

    nib.save(
        output_image,
        str(output_path),
    )

    if save_voxel_count:
        count_output_path = output_path.with_name(
            output_path.name.replace(
                "pattern_change_mean_",
                "pattern_change_trial_count_",
            )
        )

        count_header = template_image.header.copy()
        count_header.set_data_dtype(
            np.float32
        )

        count_image = nib.Nifti1Image(
            count_array.astype(np.float32),
            affine=reference_affine,
            header=count_header,
        )

        nib.save(
            count_image,
            str(count_output_path),
        )

    return (
        number_used,
        number_found,
        skipped_images,
    )


def clean_tsv_field(
    value: object,
) -> str:
    """Remove tabs and line breaks before writing a TSV field."""
    return (
        str(value)
        .replace("\t", " ")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def write_summary(
    output_root: Path,
    summary_rows: list[dict[str, object]],
) -> Path:
    """Write the participant-stage processing summary."""
    summary_path = (
        output_root
        / "stage_mean_summary.tsv"
    )

    columns = [
        "participant_id",
        "label",
        "n_files_found",
        "n_files_used",
        "n_files_skipped",
        "output_file",
        "status",
    ]

    with summary_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        file.write(
            "\t".join(columns)
            + "\n"
        )

        for row in summary_rows:
            values = [
                clean_tsv_field(
                    row.get(column, "")
                )
                for column in columns
            ]

            file.write(
                "\t".join(values)
                + "\n"
            )

    return summary_path


def write_skipped_images(
    output_root: Path,
    skipped_rows: list[dict[str, object]],
) -> Path:
    """Write a report describing trial images excluded from averaging."""
    skipped_path = (
        output_root
        / "skipped_images.tsv"
    )

    columns = [
        "participant_id",
        "label",
        "image_file",
        "reason",
    ]

    with skipped_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        file.write(
            "\t".join(columns)
            + "\n"
        )

        for row in skipped_rows:
            values = [
                clean_tsv_field(
                    row.get(column, "")
                )
                for column in columns
            ]

            file.write(
                "\t".join(values)
                + "\n"
            )

    return skipped_path


def main() -> None:
    """Create participant- and stage-specific mean images."""
    args = parse_arguments()

    input_root = (
        args.input_root
        .expanduser()
        .resolve()
    )

    output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    labels = normalize_labels(
        args.labels
    )

    if not input_root.is_dir():
        raise NotADirectoryError(
            "Input directory does not exist: "
            f"{input_root}"
        )

    if args.affine_tolerance < 0:
        raise ValueError(
            "--affine-tolerance must be zero or greater."
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    subject_directories = get_subject_directories(
        input_root,
        args.subject_ids,
    )

    if not subject_directories:
        raise RuntimeError(
            "No participant directories were found or selected."
        )

    print(
        f"Input root:          {input_root}"
    )
    print(
        f"Output root:         {output_root}"
    )
    print(
        f"Participants:        {len(subject_directories)}"
    )
    print(
        f"Labels:              {', '.join(labels)}"
    )
    print(
        f"Output extension:    {args.output_extension}"
    )
    print(
        f"Skip existing:       {args.skip_existing}"
    )
    print(
        f"Save voxel counts:   {args.save_voxel_count}"
    )

    total_outputs_written = 0
    total_stage_combinations = (
        len(subject_directories)
        * len(labels)
    )

    summary_rows: list[dict[str, object]] = []
    skipped_image_rows: list[dict[str, object]] = []

    for subject_directory in subject_directories:
        subject_name = subject_directory.name

        outputs_for_subject = 0

        for label in labels:
            label_directory = (
                subject_directory
                / label
            )

            output_label_directory = (
                output_root
                / subject_name
                / label
            )

            output_path = (
                output_label_directory
                / (
                    f"pattern_change_mean_{label}"
                    f"{args.output_extension}"
                )
            )

            if not label_directory.is_dir():
                reason = "label directory does not exist"

                print(
                    f"[Skipped] {subject_name}/{label}: "
                    f"{reason}."
                )

                summary_rows.append(
                    {
                        "participant_id": subject_name,
                        "label": label,
                        "n_files_found": 0,
                        "n_files_used": 0,
                        "n_files_skipped": 0,
                        "output_file": "",
                        "status": reason,
                    }
                )
                continue

            nifti_files = list_nifti_files(
                label_directory
            )

            if not nifti_files:
                reason = "no trial-wise NIfTI files found"

                print(
                    f"[Skipped] {subject_name}/{label}: "
                    f"{reason}."
                )

                summary_rows.append(
                    {
                        "participant_id": subject_name,
                        "label": label,
                        "n_files_found": 0,
                        "n_files_used": 0,
                        "n_files_skipped": 0,
                        "output_file": "",
                        "status": reason,
                    }
                )
                continue

            if (
                args.skip_existing
                and output_path.is_file()
            ):
                print(
                    f"[Existing] {subject_name}/{label}: "
                    f"{output_path}"
                )

                summary_rows.append(
                    {
                        "participant_id": subject_name,
                        "label": label,
                        "n_files_found": len(nifti_files),
                        "n_files_used": "",
                        "n_files_skipped": "",
                        "output_file": output_path,
                        "status": "existing output skipped",
                    }
                )
                continue

            (
                number_used,
                number_found,
                skipped_images,
            ) = compute_mean_nifti(
                nifti_paths=nifti_files,
                output_path=output_path,
                affine_tolerance=args.affine_tolerance,
                save_voxel_count=args.save_voxel_count,
            )

            for image_path, reason in skipped_images:
                skipped_image_rows.append(
                    {
                        "participant_id": subject_name,
                        "label": label,
                        "image_file": image_path,
                        "reason": reason,
                    }
                )

            if number_used > 0:
                number_skipped = (
                    number_found
                    - number_used
                )

                print(
                    f"[Completed] {subject_name}/{label}: "
                    f"used {number_used}/{number_found} images -> "
                    f"{output_path}"
                )

                total_outputs_written += 1
                outputs_for_subject += 1

                summary_rows.append(
                    {
                        "participant_id": subject_name,
                        "label": label,
                        "n_files_found": number_found,
                        "n_files_used": number_used,
                        "n_files_skipped": number_skipped,
                        "output_file": output_path,
                        "status": "completed",
                    }
                )

            else:
                print(
                    f"[Skipped] {subject_name}/{label}: "
                    "no valid images remained."
                )

                summary_rows.append(
                    {
                        "participant_id": subject_name,
                        "label": label,
                        "n_files_found": number_found,
                        "n_files_used": 0,
                        "n_files_skipped": number_found,
                        "output_file": "",
                        "status": "no valid images",
                    }
                )

        print(
            f"[Participant summary] {subject_name}: "
            f"{outputs_for_subject} stage means written."
        )

    summary_path = write_summary(
        output_root,
        summary_rows,
    )

    skipped_images_path = write_skipped_images(
        output_root,
        skipped_image_rows,
    )

    print("\n========== Final summary ==========")
    print(
        f"Participant-stage combinations: {total_stage_combinations}"
    )
    print(
        f"Mean images written:            {total_outputs_written}"
    )
    print(
        f"Skipped trial images:           {len(skipped_image_rows)}"
    )
    print(
        f"Processing summary:             {summary_path}"
    )
    print(
        f"Skipped-image report:           {skipped_images_path}"
    )
    print(
        f"Output root:                    {output_root}"
    )


if __name__ == "__main__":
    main()