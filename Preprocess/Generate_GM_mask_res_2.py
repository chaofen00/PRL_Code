#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create a group-level gray-matter mask from fMRIPrep outputs.

Workflow
--------
1. Select a preprocessed four-dimensional BOLD image as the spatial reference.
2. Derive a three-dimensional reference grid from the BOLD image.
3. Resample each participant's gray-matter probability map to that grid.
4. Average the resampled probability maps.
5. Threshold the group mean map to generate binary gray-matter masks.

Public-release version
----------------------
- Contains no user-specific drive letters or absolute paths.
- Input and output locations are supplied through command-line arguments.
- Uses relative default paths when arguments are omitted.

Dependencies
------------
pip install nibabel nilearn numpy

Example
-------
python create_group_gm_mask.py \
    --fmriprep-dir /path/to/fmriprep \
    --space MNI152NLin2009cAsym_res-2 \
    --task-name gonogo \
    --runs 01 \
    --thresholds 0.20 0.30

A specific reference BOLD image can also be supplied:

python create_group_gm_mask.py \
    --fmriprep-dir /path/to/fmriprep \
    --reference-bold /path/to/reference_bold.nii.gz
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img


def parse_arguments() -> argparse.Namespace:
    """Parse portable input, output, and analysis settings."""
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Create a group mean gray-matter probability map and "
            "thresholded gray-matter masks from fMRIPrep outputs."
        )
    )

    parser.add_argument(
        "--fmriprep-dir",
        type=Path,
        default=script_dir / "data" / "fmriprep",
        help=(
            "Root directory containing fMRIPrep participant folders. "
            "Default: <script_dir>/data/fmriprep"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. When omitted, outputs are written to "
            "<fmriprep_dir>/group_masks."
        ),
    )

    parser.add_argument(
        "--space",
        default="MNI152NLin2009cAsym_res-2",
        help=(
            "BIDS spatial entity used in the fMRIPrep filenames. "
            "Default: MNI152NLin2009cAsym_res-2"
        ),
    )

    parser.add_argument(
        "--task-name",
        default="gonogo",
        help=(
            "BIDS task name used when automatically selecting a "
            "reference BOLD image. Default: gonogo"
        ),
    )

    parser.add_argument(
        "--runs",
        nargs="+",
        default=["01"],
        help=(
            "Run labels searched when automatically selecting a "
            "reference BOLD image. Default: 01"
        ),
    )

    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.20, 0.30],
        help=(
            "Probability thresholds used to generate binary masks. "
            "Default: 0.20 0.30"
        ),
    )

    parser.add_argument(
        "--reference-bold",
        type=Path,
        default=None,
        help=(
            "Optional explicit four-dimensional BOLD reference image. "
            "When omitted, the script automatically selects one."
        ),
    )

    parser.add_argument(
        "--interpolation",
        choices=["continuous", "linear", "nearest"],
        default="continuous",
        help=(
            "Interpolation method used to resample probability maps. "
            "Default: continuous"
        ),
    )

    return parser.parse_args()


def natural_sort_key(value: str) -> list[object]:
    """Return a natural sorting key, for example sub-2 before sub-10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def all_subjects(root: Path) -> list[str]:
    """Return participant labels without the sub- prefix."""
    subjects = [
        path.name.removeprefix("sub-")
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("sub-")
    ]

    return sorted(
        subjects,
        key=natural_sort_key,
    )


def find_first(patterns: Iterable[str]) -> Path | None:
    """Return the first file matching any supplied glob pattern."""
    for pattern in patterns:
        matches = sorted(
            Path().glob(pattern),
            key=lambda path: natural_sort_key(str(path)),
        )

        if matches:
            return matches[0]

    return None


def find_first_under_root(
    root: Path,
    patterns: Iterable[str],
) -> Path | None:
    """Return the first path under root matching one of the patterns."""
    for pattern in patterns:
        matches = sorted(
            root.glob(pattern),
            key=lambda path: natural_sort_key(str(path)),
        )

        if matches:
            return matches[0]

    return None


def gm_prob_path_for_subject(
    root: Path,
    subject_id: str,
    space: str,
) -> Path | None:
    """
    Locate a participant gray-matter probability segmentation.

    Supported structures
    --------------------
    sub-XXX/anat/
    sub-XXX/ses-*/anat/
    """
    patterns = [
        (
            f"sub-{subject_id}/anat/"
            f"sub-{subject_id}_space-{space}_label-GM_probseg.nii.gz"
        ),
        (
            f"sub-{subject_id}/ses-*/anat/"
            f"sub-{subject_id}_ses-*_space-{space}_label-GM_probseg.nii.gz"
        ),
    ]

    return find_first_under_root(
        root,
        patterns,
    )


def auto_pick_reference_bold(
    root: Path,
    space: str,
    task_name: str,
    run_list: list[str],
) -> Path | None:
    """
    Automatically select the first available preprocessed BOLD image.

    Supported structures
    --------------------
    sub-XXX/func/
    sub-XXX/ses-*/func/
    """
    subject_ids = all_subjects(root)

    for subject_id in subject_ids:
        for run in run_list:
            patterns = [
                (
                    f"sub-{subject_id}/func/"
                    f"sub-{subject_id}_task-{task_name}_run-{run}_"
                    f"space-{space}_desc-preproc_bold.nii.gz"
                ),
                (
                    f"sub-{subject_id}/ses-*/func/"
                    f"sub-{subject_id}_ses-*_task-{task_name}_run-{run}_"
                    f"space-{space}_desc-preproc_bold.nii.gz"
                ),
            ]

            reference_path = find_first_under_root(
                root,
                patterns,
            )

            if reference_path is not None:
                return reference_path

    return None


def make_3d_reference_from_4d_bold(
    bold_image: nib.spatialimages.SpatialImage,
) -> nib.Nifti1Image:
    """
    Create a three-dimensional reference grid from a 4D BOLD image.

    The returned image shares the first three spatial dimensions, affine,
    and relevant header information with the BOLD image.
    """
    if len(bold_image.shape) != 4:
        raise ValueError(
            "The reference BOLD image must be four-dimensional. "
            f"Received shape: {bold_image.shape}"
        )

    x_size, y_size, z_size, _ = bold_image.shape

    header = bold_image.header.copy()
    header.set_data_dtype(np.float32)

    reference_data = np.zeros(
        (
            x_size,
            y_size,
            z_size,
        ),
        dtype=np.float32,
    )

    return nib.Nifti1Image(
        reference_data,
        bold_image.affine,
        header,
    )


def save_like(
    reference_image: nib.spatialimages.SpatialImage,
    data: np.ndarray,
    output_path: Path,
    dtype: np.dtype = np.float32,
) -> None:
    """Save a three-dimensional array using a reference grid."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    header = reference_image.header.copy()

    if np.issubdtype(dtype, np.integer):
        output_dtype = np.uint8
    else:
        output_dtype = np.float32

    header.set_data_dtype(output_dtype)

    output_image = nib.Nifti1Image(
        np.asarray(
            data,
            dtype=output_dtype,
        ),
        reference_image.affine,
        header,
    )

    nib.save(
        output_image,
        str(output_path),
    )


def validate_thresholds(
    thresholds: list[float],
) -> list[float]:
    """Validate and sort probability thresholds."""
    if not thresholds:
        raise ValueError(
            "At least one gray-matter probability threshold is required."
        )

    invalid_thresholds = [
        threshold
        for threshold in thresholds
        if not 0.0 <= threshold <= 1.0
    ]

    if invalid_thresholds:
        raise ValueError(
            "Probability thresholds must lie between 0 and 1. "
            f"Invalid values: {invalid_thresholds}"
        )

    return sorted(set(thresholds))


def main() -> None:
    """Create the group mean GM probability map and binary masks."""
    args = parse_arguments()

    fmriprep_dir = (
        args.fmriprep_dir
        .expanduser()
        .resolve()
    )

    if args.output_dir is None:
        output_dir = (
            fmriprep_dir
            / "group_masks"
        )
    else:
        output_dir = (
            args.output_dir
            .expanduser()
            .resolve()
        )

    thresholds = validate_thresholds(
        args.thresholds
    )

    if not fmriprep_dir.is_dir():
        raise NotADirectoryError(
            "The fMRIPrep directory does not exist: "
            f"{fmriprep_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. Select reference BOLD and derive a 3D reference grid
    # ========================================================

    if args.reference_bold is None:
        reference_bold_path = auto_pick_reference_bold(
            root=fmriprep_dir,
            space=args.space,
            task_name=args.task_name,
            run_list=args.runs,
        )

        if reference_bold_path is None:
            raise RuntimeError(
                "No suitable reference BOLD image was found. "
                "Provide one explicitly using --reference-bold."
            )

    else:
        reference_bold_path = (
            args.reference_bold
            .expanduser()
            .resolve()
        )

        if not reference_bold_path.is_file():
            raise FileNotFoundError(
                "The reference BOLD image does not exist: "
                f"{reference_bold_path}"
            )

    print(
        "Reference BOLD image: "
        f"{reference_bold_path}"
    )

    bold_4d_image = nib.load(
        str(reference_bold_path)
    )

    reference_3d_image = (
        make_3d_reference_from_4d_bold(
            bold_4d_image
        )
    )

    reference_shape = (
        reference_3d_image.shape
    )

    print(
        "Reference 3D grid shape: "
        f"{reference_shape}"
    )

    print(
        "Reference affine:\n"
        f"{reference_3d_image.affine}"
    )

    # ========================================================
    # 2. Find participants and resample GM probability maps
    # ========================================================

    subject_ids = all_subjects(
        fmriprep_dir
    )

    if not subject_ids:
        raise RuntimeError(
            "No participant directories matching sub-* were found in: "
            f"{fmriprep_dir}"
        )

    probability_sum = np.zeros(
        reference_shape,
        dtype=np.float64,
    )

    number_used = 0
    included_subjects: list[str] = []
    excluded_subjects: list[tuple[str, str]] = []

    for subject_id in subject_ids:
        gm_probability_path = (
            gm_prob_path_for_subject(
                root=fmriprep_dir,
                subject_id=subject_id,
                space=args.space,
            )
        )

        if gm_probability_path is None:
            message = "GM probability segmentation not found"
            excluded_subjects.append(
                (
                    subject_id,
                    message,
                )
            )

            print(
                f"[sub-{subject_id}] skipped: {message}."
            )
            continue

        try:
            gm_probability_image = nib.load(
                str(gm_probability_path)
            )

            resampled_image = resample_to_img(
                source_img=gm_probability_image,
                target_img=reference_3d_image,
                interpolation=args.interpolation,
            )

            resampled_probability = (
                resampled_image.get_fdata(
                    dtype=np.float32
                )
            )

            if resampled_probability.ndim != 3:
                raise ValueError(
                    "Resampled GM probability image is not 3D: "
                    f"shape={resampled_probability.shape}"
                )

            if (
                resampled_probability.shape
                != reference_shape
            ):
                raise ValueError(
                    "Resampled GM image does not match the "
                    "reference shape: "
                    f"{resampled_probability.shape} "
                    f"versus {reference_shape}"
                )

            if not np.all(
                np.isfinite(
                    resampled_probability
                )
            ):
                number_nonfinite = int(
                    np.size(resampled_probability)
                    - np.count_nonzero(
                        np.isfinite(
                            resampled_probability
                        )
                    )
                )

                print(
                    f"[sub-{subject_id}] warning: "
                    f"{number_nonfinite} non-finite values "
                    "were replaced with zero."
                )

                resampled_probability = (
                    np.nan_to_num(
                        resampled_probability,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                )

            # Probability values should lie between zero and one.
            resampled_probability = np.clip(
                resampled_probability,
                0.0,
                1.0,
            )

            probability_sum += (
                resampled_probability
            )

            number_used += 1
            included_subjects.append(
                subject_id
            )

            print(
                f"[sub-{subject_id}] included: "
                f"{gm_probability_path.name}"
            )

        except Exception as error:
            excluded_subjects.append(
                (
                    subject_id,
                    str(error),
                )
            )

            print(
                f"[sub-{subject_id}] skipped due to error: "
                f"{error}"
            )

    if number_used == 0:
        raise RuntimeError(
            "No participants were included because no valid "
            "gray-matter probability maps were found."
        )

    # ========================================================
    # 3. Calculate and save group mean GM probability
    # ========================================================

    mean_probability = (
        probability_sum
        / float(number_used)
    ).astype(np.float32)

    mean_probability_filename = (
        f"group_mean_GMprob_"
        f"{args.space}_"
        f"boldres3D_"
        f"n{number_used}.nii.gz"
    )

    mean_probability_path = (
        output_dir
        / mean_probability_filename
    )

    save_like(
        reference_image=reference_3d_image,
        data=mean_probability,
        output_path=mean_probability_path,
        dtype=np.float32,
    )

    print(
        "Saved group mean GM probability map: "
        f"{mean_probability_path}"
    )

    # ========================================================
    # 4. Threshold group mean probability to create masks
    # ========================================================

    for threshold in thresholds:
        binary_mask = (
            mean_probability > threshold
        ).astype(np.uint8)

        mask_filename = (
            f"group_GMmask_"
            f"p{threshold:.2f}_"
            f"{args.space}_"
            f"boldres3D_"
            f"n{number_used}.nii.gz"
        )

        mask_path = (
            output_dir
            / mask_filename
        )

        save_like(
            reference_image=reference_3d_image,
            data=binary_mask,
            output_path=mask_path,
            dtype=np.uint8,
        )

        number_of_mask_voxels = int(
            binary_mask.sum()
        )

        print(
            f"Saved group mask at p > {threshold:.2f}: "
            f"{mask_path}"
        )

        print(
            f"Mask voxels at p > {threshold:.2f}: "
            f"{number_of_mask_voxels}"
        )

    # ========================================================
    # 5. Save participant inclusion and exclusion records
    # ========================================================

    included_subjects_path = (
        output_dir
        / "included_subjects.txt"
    )

    with included_subjects_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for subject_id in included_subjects:
            file.write(
                f"sub-{subject_id}\n"
            )

    excluded_subjects_path = (
        output_dir
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
            clean_reason = (
                str(reason)
                .replace("\t", " ")
                .replace("\n", " ")
            )

            file.write(
                f"sub-{subject_id}\t{clean_reason}\n"
            )

    reference_record_path = (
        output_dir
        / "reference_bold.txt"
    )

    with reference_record_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            f"{reference_bold_path}\n"
        )

    print(
        f"Included participants: {number_used}"
    )

    print(
        f"Excluded participants: {len(excluded_subjects)}"
    )

    print(
        "Group gray-matter mask generation completed."
    )


if __name__ == "__main__":
    main()