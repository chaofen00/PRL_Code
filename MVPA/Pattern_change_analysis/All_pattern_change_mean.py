#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calculate participant-level and group-level mean pattern-change maps.

Workflow
--------
1. Locate each participant's trial-wise pattern-change images.
2. Calculate a voxel-wise mean across trials while ignoring non-finite values.
3. Save one participant-level mean image per participant.
4. Calculate a voxel-wise group mean across participant-level mean images.
5. Save a voxel-wise participant-count image for quality control.

Public-release version
----------------------
- Contains no local drive letters, user names, or machine-specific paths.
- Input and output directories are supplied through command-line arguments.
- Participant directories can be detected automatically.
- Both .nii and .nii.gz trial images are supported.

Expected input structure
------------------------
data/
└── pattern_change/
    ├── Sub02_Pattern_change/
    │   ├── pattern_change_trial_001.nii
    │   ├── pattern_change_trial_002.nii
    │   └── ...
    ├── Sub03_Pattern_change/
    │   └── ...
    └── ...

Default output structure
------------------------
data/
└── pattern_change/
    └── _means/
        ├── Sub02_pattern_change_mean.nii
        ├── Sub03_pattern_change_mean.nii
        ├── Group_pattern_change_mean.nii
        ├── Group_subject_count.nii
        ├── included_subjects.txt
        └── subject_trial_counts.tsv

Dependencies
------------
pip install nibabel nilearn numpy

Example
-------
python calculate_pattern_change_means.py \
    --base-output-dir /path/to/pattern_change \
    --average-output-dir /path/to/pattern_change_means

Process selected participants only
----------------------------------
python calculate_pattern_change_means.py \
    --base-output-dir /path/to/pattern_change \
    --subject-ids 02 03 04 05
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
from nilearn.image import load_img, new_img_like


TRIAL_PATTERN = re.compile(
    r"^pattern_change_trial_(\d+)\.nii(?:\.gz)?$",
    flags=re.IGNORECASE,
)

SUBJECT_DIRECTORY_PATTERN = re.compile(
    r"^Sub(\d+)_Pattern_change$",
    flags=re.IGNORECASE,
)


def parse_arguments() -> argparse.Namespace:
    """Parse portable input and output settings."""
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Calculate participant-level and group-level mean "
            "pattern-change images."
        )
    )

    parser.add_argument(
        "--base-output-dir",
        type=Path,
        default=script_dir / "data" / "pattern_change",
        help=(
            "Root directory containing SubXX_Pattern_change folders. "
            "Default: <script_dir>/data/pattern_change"
        ),
    )

    parser.add_argument(
        "--average-output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for participant and group mean images. "
            "When omitted, <base-output-dir>/_means is used."
        ),
    )

    parser.add_argument(
        "--subject-ids",
        nargs="+",
        default=None,
        help=(
            "Optional participant identifiers, such as 02 03 04. "
            "When omitted, participants are detected automatically."
        ),
    )

    parser.add_argument(
        "--affine-tolerance",
        type=float,
        default=1e-5,
        help=(
            "Absolute tolerance used when comparing NIfTI affines. "
            "Default: 1e-5"
        ),
    )

    parser.add_argument(
        "--output-extension",
        choices=[".nii", ".nii.gz"],
        default=".nii",
        help="Output image extension. Default: .nii",
    )

    parser.add_argument(
        "--no-group-count",
        action="store_true",
        help="Do not save the voxel-wise participant-count image.",
    )

    return parser.parse_args()


def natural_sort_key(value: str) -> list[object]:
    """Sort text naturally, for example Sub2 before Sub10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def normalize_subject_id(subject_id: str) -> str:
    """
    Normalize a participant identifier.

    Examples
    --------
    Sub02 -> 02
    sub-002 -> 002
    2 -> 2
    """
    match = re.search(
        r"(\d+)",
        str(subject_id),
    )

    if match is None:
        raise ValueError(
            f"Could not extract a numeric participant ID from: {subject_id}"
        )

    return match.group(1)


def detect_subject_ids(
    base_output_dir: Path,
) -> list[str]:
    """Detect participant IDs from SubXX_Pattern_change directories."""
    detected: list[tuple[str, str]] = []

    for path in base_output_dir.iterdir():
        if not path.is_dir():
            continue

        match = SUBJECT_DIRECTORY_PATTERN.fullmatch(
            path.name
        )

        if match is None:
            continue

        subject_id = match.group(1)

        detected.append(
            (
                subject_id,
                path.name,
            )
        )

    detected.sort(
        key=lambda item: natural_sort_key(
            item[1]
        )
    )

    return [
        subject_id
        for subject_id, _ in detected
    ]


def get_subject_ids(
    base_output_dir: Path,
    requested_subject_ids: Sequence[str] | None,
) -> list[str]:
    """Return requested IDs or automatically detected IDs."""
    if requested_subject_ids is None:
        subject_ids = detect_subject_ids(
            base_output_dir
        )
    else:
        subject_ids = [
            normalize_subject_id(subject_id)
            for subject_id in requested_subject_ids
        ]

    subject_ids = sorted(
        set(subject_ids),
        key=natural_sort_key,
    )

    if not subject_ids:
        raise RuntimeError(
            "No participant directories were detected and no "
            "participant IDs were supplied."
        )

    return subject_ids


def find_subject_directory(
    base_output_dir: Path,
    subject_id: str,
) -> Path | None:
    """
    Locate a participant directory while preserving common zero-padding forms.

    Supported examples
    ------------------
    Sub02_Pattern_change
    Sub002_Pattern_change
    Sub2_Pattern_change
    """
    participant_number = int(
        subject_id
    )

    candidate_names = [
        f"Sub{subject_id}_Pattern_change",
        f"Sub{participant_number:02d}_Pattern_change",
        f"Sub{participant_number:03d}_Pattern_change",
        f"Sub{participant_number}_Pattern_change",
    ]

    checked_names: set[str] = set()

    for candidate_name in candidate_names:
        if candidate_name in checked_names:
            continue

        checked_names.add(
            candidate_name
        )

        candidate_path = (
            base_output_dir
            / candidate_name
        )

        if candidate_path.is_dir():
            return candidate_path

    return None


def list_trial_files(
    subject_directory: Path,
) -> list[Path]:
    """Return trial-wise pattern-change files sorted by trial number."""
    trial_files: list[tuple[int, Path]] = []

    if not subject_directory.is_dir():
        return []

    for path in subject_directory.iterdir():
        if not path.is_file():
            continue

        match = TRIAL_PATTERN.fullmatch(
            path.name
        )

        if match is None:
            continue

        trial_number = int(
            match.group(1)
        )

        trial_files.append(
            (
                trial_number,
                path,
            )
        )

    trial_files.sort(
        key=lambda item: item[0]
    )

    return [
        path
        for _, path in trial_files
    ]


def same_grid(
    image_a: nib.spatialimages.SpatialImage,
    image_b: nib.spatialimages.SpatialImage,
    affine_tolerance: float,
) -> bool:
    """Check whether two images share spatial dimensions and affine."""
    return (
        tuple(image_a.shape[:3])
        == tuple(image_b.shape[:3])
        and np.allclose(
            image_a.affine,
            image_b.affine,
            atol=affine_tolerance,
            rtol=0.0,
        )
    )


def require_same_grid(
    reference_image: nib.spatialimages.SpatialImage,
    current_image: nib.spatialimages.SpatialImage,
    reference_path: Path,
    current_path: Path,
    affine_tolerance: float,
) -> None:
    """Raise an error when two NIfTI images do not share the same grid."""
    if same_grid(
        reference_image,
        current_image,
        affine_tolerance,
    ):
        return

    raise ValueError(
        "NIfTI grid mismatch.\n"
        f"Reference file: {reference_path}\n"
        f"Reference shape: {reference_image.shape}\n"
        f"Current file: {current_path}\n"
        f"Current shape: {current_image.shape}"
    )


def streaming_finite_mean_from_files(
    nifti_paths: Sequence[Path],
    affine_tolerance: float,
) -> tuple[
    np.ndarray | None,
    nib.spatialimages.SpatialImage | None,
    np.ndarray | None,
]:
    """
    Calculate a voxel-wise mean while ignoring NaN and Inf.

    The calculation is performed incrementally:

        mean = sum(valid values) / count(valid values)

    Returns
    -------
    mean_array:
        Voxel-wise mean image.
    template_image:
        First valid input image, used as the output template.
    count_array:
        Number of valid trial values contributing at each voxel.
    """
    template_image = None
    template_path = None

    sum_array = None
    count_array = None

    for nifti_path in nifti_paths:
        image = load_img(
            str(nifti_path)
        )

        if len(image.shape) != 3:
            raise ValueError(
                "Trial-wise pattern-change images must be 3D. "
                f"File: {nifti_path}; shape: {image.shape}"
            )

        if template_image is None:
            template_image = image
            template_path = nifti_path

            sum_array = np.zeros(
                image.shape,
                dtype=np.float64,
            )

            count_array = np.zeros(
                image.shape,
                dtype=np.uint32,
            )

        else:
            require_same_grid(
                template_image,
                image,
                template_path,
                nifti_path,
                affine_tolerance,
            )

        data = image.get_fdata(
            dtype=np.float32
        )

        valid_values = np.isfinite(
            data
        )

        sum_array[valid_values] += (
            data[valid_values]
        )

        count_array[valid_values] += 1

    if template_image is None:
        return None, None, None

    mean_array = np.full(
        sum_array.shape,
        np.nan,
        dtype=np.float32,
    )

    np.divide(
        sum_array,
        count_array,
        out=mean_array,
        where=count_array > 0,
    )

    return (
        mean_array,
        template_image,
        count_array,
    )


def save_image_like(
    template_image: nib.spatialimages.SpatialImage,
    data: np.ndarray,
    output_path: Path,
    dtype: np.dtype = np.float32,
) -> None:
    """Save data using the grid and header of a template image."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_data = np.asarray(
        data,
        dtype=dtype,
    )

    output_image = new_img_like(
        template_image,
        output_data,
        copy_header=True,
    )

    output_image.to_filename(
        str(output_path)
    )


def clean_text_field(value: object) -> str:
    """Remove tabs and line breaks before writing a TSV field."""
    return (
        str(value)
        .replace("\t", " ")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def main() -> None:
    """Calculate participant-level and group-level pattern-change means."""
    args = parse_arguments()

    base_output_dir = (
        args.base_output_dir
        .expanduser()
        .resolve()
    )

    if args.average_output_dir is None:
        average_output_dir = (
            base_output_dir
            / "_means"
        )
    else:
        average_output_dir = (
            args.average_output_dir
            .expanduser()
            .resolve()
        )

    if not base_output_dir.is_dir():
        raise NotADirectoryError(
            "Pattern-change root directory does not exist: "
            f"{base_output_dir}"
        )

    if args.affine_tolerance < 0:
        raise ValueError(
            "--affine-tolerance must be zero or greater."
        )

    average_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    subject_ids = get_subject_ids(
        base_output_dir,
        args.subject_ids,
    )

    print(
        "Pattern-change root: "
        f"{base_output_dir}"
    )

    print(
        "Mean-image output directory: "
        f"{average_output_dir}"
    )

    print(
        "Participants scheduled: "
        f"{len(subject_ids)}"
    )

    group_sum = None
    group_count = None

    group_template = None
    group_template_path = None

    finished_subjects: list[str] = []
    excluded_subjects: list[tuple[str, str]] = []
    subject_trial_counts: list[tuple[str, int]] = []

    for subject_id in subject_ids:
        subject_directory = find_subject_directory(
            base_output_dir,
            subject_id,
        )

        if subject_directory is None:
            reason = "participant directory not found"

            excluded_subjects.append(
                (
                    subject_id,
                    reason,
                )
            )

            print(
                f"[Skipped] Sub{subject_id}: {reason}."
            )
            continue

        trial_files = list_trial_files(
            subject_directory
        )

        if not trial_files:
            reason = (
                "no pattern_change_trial_*.nii or "
                "pattern_change_trial_*.nii.gz files found"
            )

            excluded_subjects.append(
                (
                    subject_id,
                    reason,
                )
            )

            print(
                f"[Skipped] Sub{subject_id}: {reason}."
            )
            continue

        try:
            (
                subject_mean,
                subject_template,
                subject_count,
            ) = streaming_finite_mean_from_files(
                trial_files,
                args.affine_tolerance,
            )

        except Exception as error:
            excluded_subjects.append(
                (
                    subject_id,
                    str(error),
                )
            )

            print(
                f"[Skipped] Sub{subject_id}: {error}"
            )
            continue

        if (
            subject_mean is None
            or subject_template is None
            or subject_count is None
        ):
            reason = "no valid trial data"

            excluded_subjects.append(
                (
                    subject_id,
                    reason,
                )
            )

            print(
                f"[Skipped] Sub{subject_id}: {reason}."
            )
            continue

        subject_output_name = (
            f"Sub{subject_id}_"
            f"pattern_change_mean"
            f"{args.output_extension}"
        )

        subject_mean_path = (
            average_output_dir
            / subject_output_name
        )

        save_image_like(
            template_image=subject_template,
            data=subject_mean,
            output_path=subject_mean_path,
            dtype=np.float32,
        )

        number_of_trials = len(
            trial_files
        )

        subject_trial_counts.append(
            (
                subject_id,
                number_of_trials,
            )
        )

        print(
            f"[OK] Sub{subject_id}: "
            f"{number_of_trials} trial images averaged -> "
            f"{subject_mean_path}"
        )

        if group_sum is None:
            group_sum = np.zeros(
                subject_mean.shape,
                dtype=np.float64,
            )

            group_count = np.zeros(
                subject_mean.shape,
                dtype=np.uint32,
            )

            group_template = subject_template
            group_template_path = subject_mean_path

        else:
            require_same_grid(
                group_template,
                subject_template,
                group_template_path,
                subject_mean_path,
                args.affine_tolerance,
            )

        valid_subject_values = np.isfinite(
            subject_mean
        )

        group_sum[valid_subject_values] += (
            subject_mean[valid_subject_values]
        )

        group_count[valid_subject_values] += 1

        finished_subjects.append(
            subject_id
        )

    if (
        group_sum is None
        or group_count is None
        or group_template is None
        or not finished_subjects
    ):
        raise RuntimeError(
            "No participant-level mean images were generated; "
            "the group mean cannot be calculated."
        )

    group_mean = np.full(
        group_sum.shape,
        np.nan,
        dtype=np.float32,
    )

    np.divide(
        group_sum,
        group_count,
        out=group_mean,
        where=group_count > 0,
    )

    group_mean_path = (
        average_output_dir
        / (
            "Group_pattern_change_mean"
            f"{args.output_extension}"
        )
    )

    save_image_like(
        template_image=group_template,
        data=group_mean,
        output_path=group_mean_path,
        dtype=np.float32,
    )

    group_count_path = None

    if not args.no_group_count:
        group_count_path = (
            average_output_dir
            / (
                "Group_subject_count"
                f"{args.output_extension}"
            )
        )

        save_image_like(
            template_image=group_template,
            data=group_count.astype(np.float32),
            output_path=group_count_path,
            dtype=np.float32,
        )

    included_subjects_path = (
        average_output_dir
        / "included_subjects.txt"
    )

    with included_subjects_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for subject_id in finished_subjects:
            file.write(
                f"Sub{subject_id}\n"
            )

    subject_trial_counts_path = (
        average_output_dir
        / "subject_trial_counts.tsv"
    )

    with subject_trial_counts_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        file.write(
            "participant_id\tn_trial_images\n"
        )

        for subject_id, number_of_trials in subject_trial_counts:
            file.write(
                f"Sub{subject_id}\t{number_of_trials}\n"
            )

    excluded_subjects_path = (
        average_output_dir
        / "excluded_subjects.tsv"
    )

    with excluded_subjects_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        file.write(
            "participant_id\treason\n"
        )

        for subject_id, reason in excluded_subjects:
            file.write(
                f"Sub{subject_id}\t"
                f"{clean_text_field(reason)}\n"
            )

    print("\n========== Group summary ==========")

    print(
        "Participants included: "
        f"{len(finished_subjects)} / {len(subject_ids)}"
    )

    print(
        "Group mean image: "
        f"{group_mean_path}"
    )

    if group_count_path is not None:
        print(
            "Voxel-wise participant-count image: "
            f"{group_count_path}"
        )

    print(
        "Included-participant record: "
        f"{included_subjects_path}"
    )

    print(
        "Trial-count record: "
        f"{subject_trial_counts_path}"
    )

    print(
        "Output directory: "
        f"{average_output_dir}"
    )


if __name__ == "__main__":
    main()