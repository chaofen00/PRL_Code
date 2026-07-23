#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Voxel-wise regression of trial-wise dual-model policy change on pattern
change.

For each participant, this script fits the following model separately at
each voxel:

    delta_policy_dual[t]
        = intercept
        + beta_voxel * pattern_change[t, voxel]

The resulting beta coefficient is saved as a participant-level NIfTI map.

No temporal detrending or z-scoring is applied.

Trial alignment
---------------
The trial number encoded in:

    pattern_change_trial_003.nii

is matched to:

    global_trial == 3

in the corresponding dual-model trial-wise CSV file.

Expected input structure
------------------------
project/
├── run_patternchange_to_delta_policy_dual.py
├── data/
│   ├── pattern_change/
│   │   ├── Sub02_Pattern_change/
│   │   │   ├── pattern_change_trial_001.nii
│   │   │   ├── pattern_change_trial_002.nii
│   │   │   └── ...
│   │   ├── Sub03_Pattern_change/
│   │   │   └── ...
│   │   └── ...
│   └── qlearning_trialwise/
│       ├── Qlearning_single_dual_trialwise_Sub02.csv
│       ├── Qlearning_single_dual_trialwise_Sub03.csv
│       └── ...
└── outputs/
    └── patternchange_to_delta_policy_dual/
        ├── Sub02_regression_patternchange2delta_policy_dual.nii
        ├── Sub03_regression_patternchange2delta_policy_dual.nii
        ├── alignment_tables/
        └── processing_summary.tsv

Required CSV columns
--------------------
global_trial
delta_policy_dual

Both .nii and .nii.gz pattern-change images are supported.

Dependencies
------------
numpy
pandas
nibabel

Example
-------
python run_patternchange_to_delta_policy_dual.py \
    --pattern-root /path/to/pattern_change \
    --csv-root /path/to/qlearning_trialwise \
    --output-root /path/to/output

Process selected participants
-----------------------------
python run_patternchange_to_delta_policy_dual.py \
    --subject-ids 02 03 04 05
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import pandas as pd


TARGET_COLUMN = "delta_policy_dual"

CSV_FILE_PATTERN = re.compile(
    r"^Qlearning_single_dual_trialwise_(Sub\d+)\.csv$",
    flags=re.IGNORECASE,
)

PATTERN_FILE_PATTERN = re.compile(
    r"^pattern_change_trial_(\d+)\.nii(?:\.gz)?$",
    flags=re.IGNORECASE,
)

EPSILON = 1e-8


def parse_arguments() -> argparse.Namespace:
    """Parse portable input, output, and regression settings."""
    script_directory = Path(__file__).resolve().parent
    default_data_root = script_directory / "data"

    parser = argparse.ArgumentParser(
        description=(
            "Regress trial-wise dual-model policy change on voxel-wise "
            "pattern-change values."
        )
    )

    parser.add_argument(
        "--pattern-root",
        type=Path,
        default=default_data_root / "pattern_change",
        help=(
            "Root directory containing SubXX_Pattern_change folders. "
            "Default: <script_dir>/data/pattern_change"
        ),
    )

    parser.add_argument(
        "--csv-root",
        type=Path,
        default=default_data_root / "qlearning_trialwise",
        help=(
            "Directory containing "
            "Qlearning_single_dual_trialwise_SubXX.csv files. "
            "Default: <script_dir>/data/qlearning_trialwise"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            script_directory
            / "outputs"
            / "patternchange_to_delta_policy_dual"
        ),
        help=(
            "Output directory for participant-level beta maps. "
            "Default: "
            "<script_dir>/outputs/patternchange_to_delta_policy_dual"
        ),
    )

    parser.add_argument(
        "--subject-ids",
        nargs="+",
        default=None,
        help=(
            "Optional participant identifiers, such as 02 03 04. "
            "When omitted, participants are detected from CSV filenames."
        ),
    )

    parser.add_argument(
        "--minimum-trials",
        type=int,
        default=6,
        help=(
            "Minimum number of aligned trials required for one "
            "participant-level analysis. Default: 6"
        ),
    )

    parser.add_argument(
        "--minimum-valid-observations",
        type=int,
        default=5,
        help=(
            "Minimum number of finite trial observations required at "
            "each voxel. Default: 5"
        ),
    )

    parser.add_argument(
        "--minimum-valid-fraction",
        type=float,
        default=0.5,
        help=(
            "Minimum fraction of aligned trials required at each voxel. "
            "The effective threshold is the maximum of this fraction and "
            "--minimum-valid-observations. Default: 0.5"
        ),
    )

    parser.add_argument(
        "--affine-tolerance",
        type=float,
        default=1e-5,
        help=(
            "Absolute tolerance used when comparing NIfTI affine "
            "matrices. Default: 1e-5"
        ),
    )

    parser.add_argument(
        "--output-extension",
        choices=(".nii", ".nii.gz"),
        default=".nii",
        help="Output image extension. Default: .nii",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip a participant when the output beta map already exists "
            "and can be read."
        ),
    )

    parser.add_argument(
        "--save-alignment-tables",
        action="store_true",
        help=(
            "Save a TSV table listing the aligned trials included for "
            "each participant."
        ),
    )

    return parser.parse_args()


def natural_sort_key(value: str) -> list[object]:
    """Sort text naturally, for example Sub2 before Sub10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def normalize_subject_id(value: str) -> str:
    """Extract the numeric component of a participant identifier."""
    match = re.search(
        r"(\d+)",
        str(value),
    )

    if match is None:
        raise ValueError(
            f"Could not parse participant ID from: {value}"
        )

    return match.group(1)


def subject_number(value: str) -> int:
    """Return the numeric participant identifier."""
    return int(
        normalize_subject_id(value)
    )


def clean_tsv_field(value: object) -> str:
    """Remove tabs and line breaks before writing TSV output."""
    return (
        str(value)
        .replace("\t", " ")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def relative_path_text(
    path: Path,
    root: Path,
) -> str:
    """Return a portable path relative to a specified root."""
    try:
        return (
            path.resolve()
            .relative_to(root.resolve())
            .as_posix()
        )

    except ValueError:
        return path.name


def read_csv_with_fallback(
    csv_path: Path,
) -> pd.DataFrame:
    """Read a CSV file using common text encodings."""
    dataframe = None
    last_error: Exception | None = None

    for encoding in (
        "utf-8-sig",
        "utf-8",
        "gb18030",
    ):
        try:
            dataframe = pd.read_csv(
                csv_path,
                encoding=encoding,
            )
            break

        except UnicodeDecodeError as error:
            last_error = error

    if dataframe is None:
        raise RuntimeError(
            f"Could not decode CSV file {csv_path.name}. "
            f"Last error: {last_error}"
        )

    return dataframe


def discover_trialwise_csv_files(
    csv_root: Path,
    requested_subject_ids: Sequence[str] | None,
) -> list[tuple[str, Path]]:
    """
    Find participant-level dual-model trial-wise CSV files.

    Returns
    -------
    records:
        List of (participant_name, csv_path) tuples.
    """
    requested_numbers = None

    if requested_subject_ids is not None:
        requested_numbers = {
            subject_number(value)
            for value in requested_subject_ids
        }

    records_by_subject: dict[
        int,
        tuple[str, Path],
    ] = {}

    for csv_path in csv_root.iterdir():
        if not csv_path.is_file():
            continue

        match = CSV_FILE_PATTERN.fullmatch(
            csv_path.name
        )

        if match is None:
            continue

        subject_name = match.group(1)
        numeric_id = subject_number(
            subject_name
        )

        if (
            requested_numbers is not None
            and numeric_id not in requested_numbers
        ):
            continue

        if numeric_id in records_by_subject:
            previous_path = records_by_subject[
                numeric_id
            ][1]

            raise RuntimeError(
                "Multiple trial-wise CSV files were found for "
                f"participant {numeric_id}: "
                f"{previous_path.name}, {csv_path.name}"
            )

        subject_digits = normalize_subject_id(
            subject_name
        )

        records_by_subject[
            numeric_id
        ] = (
            f"Sub{subject_digits}",
            csv_path,
        )

    records = [
        records_by_subject[key]
        for key in sorted(
            records_by_subject
        )
    ]

    if requested_numbers is not None:
        found_numbers = set(
            records_by_subject
        )

        missing_numbers = sorted(
            requested_numbers
            - found_numbers
        )

        for missing_number in missing_numbers:
            print(
                "[Missing] No trial-wise CSV file was found for "
                f"participant {missing_number}."
            )

    return records


def find_pattern_directory(
    pattern_root: Path,
    subject_name: str,
) -> Path | None:
    """
    Locate a participant pattern-change directory.

    Supported examples
    ------------------
    Sub2_Pattern_change
    Sub02_Pattern_change
    Sub002_Pattern_change
    """
    subject_id = normalize_subject_id(
        subject_name
    )

    numeric_id = int(
        subject_id
    )

    subject_variants = [
        subject_name,
        f"Sub{subject_id}",
        f"Sub{numeric_id}",
        f"Sub{numeric_id:02d}",
        f"Sub{numeric_id:03d}",
    ]

    suffix_variants = (
        "_Pattern_change",
        "_pattern_change",
    )

    checked_paths: set[Path] = set()

    for subject_variant in subject_variants:
        for suffix in suffix_variants:
            candidate_path = (
                pattern_root
                / f"{subject_variant}{suffix}"
            )

            if candidate_path in checked_paths:
                continue

            checked_paths.add(
                candidate_path
            )

            if candidate_path.is_dir():
                return candidate_path

    return None


def list_pattern_change_files(
    pattern_directory: Path,
) -> dict[int, Path]:
    """
    Return pattern-change files indexed by global trial number.

    When both .nii and .nii.gz versions of the same trial exist, the
    uncompressed .nii file is preferred.
    """
    files_by_trial: dict[int, Path] = {}

    for image_path in pattern_directory.iterdir():
        if not image_path.is_file():
            continue

        match = PATTERN_FILE_PATTERN.fullmatch(
            image_path.name
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
            files_by_trial[
                trial_number
            ] = image_path
            continue

        if (
            image_path.name.lower().endswith(".nii")
            and existing_path.name.lower().endswith(".nii.gz")
        ):
            files_by_trial[
                trial_number
            ] = image_path

    return dict(
        sorted(
            files_by_trial.items(),
            key=lambda item: item[0],
        )
    )


def prepare_dual_policy_table(
    dataframe: pd.DataFrame,
    subject_name: str,
) -> pd.DataFrame:
    """
    Validate global trials and delta_policy_dual values.

    Rows containing missing or non-numeric values are excluded. When a
    global trial is duplicated, the final occurrence is retained.
    """
    required_columns = {
        "global_trial",
        TARGET_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "The CSV lacks required columns: "
            f"{sorted(missing_columns)}"
        )

    policy_table = dataframe[
        [
            "global_trial",
            TARGET_COLUMN,
        ]
    ].copy()

    policy_table[
        "global_trial"
    ] = pd.to_numeric(
        policy_table[
            "global_trial"
        ],
        errors="coerce",
    )

    policy_table[
        TARGET_COLUMN
    ] = pd.to_numeric(
        policy_table[
            TARGET_COLUMN
        ],
        errors="coerce",
    )

    policy_table = policy_table.dropna(
        subset=[
            "global_trial",
            TARGET_COLUMN,
        ]
    ).copy()

    policy_table[
        "global_trial"
    ] = policy_table[
        "global_trial"
    ].astype(int)

    policy_table = policy_table.loc[
        policy_table[
            "global_trial"
        ] >= 1
    ].copy()

    duplicated_trials = policy_table.loc[
        policy_table[
            "global_trial"
        ].duplicated(
            keep=False
        ),
        "global_trial",
    ].unique()

    if duplicated_trials.size > 0:
        displayed_trials = ", ".join(
            str(int(value))
            for value in duplicated_trials[:10]
        )

        print(
            f"[Warning] {subject_name}: duplicated global_trial values "
            f"were found ({displayed_trials}). "
            "The last occurrence will be retained."
        )

        policy_table = policy_table.drop_duplicates(
            subset=[
                "global_trial",
            ],
            keep="last",
        )

    policy_table = policy_table.sort_values(
        by="global_trial"
    ).reset_index(
        drop=True
    )

    if policy_table.empty:
        raise ValueError(
            "No finite delta_policy_dual values remained after filtering."
        )

    return policy_table


def load_3d_image(
    image_path: Path,
) -> tuple[
    nib.spatialimages.SpatialImage,
    np.ndarray,
]:
    """
    Load a NIfTI image as a three-dimensional float32 array.

    Singleton four-dimensional images are reduced to three dimensions.
    """
    image = nib.load(
        str(image_path)
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
            f"Expected a 3D image, received shape {data.shape}: "
            f"{image_path.name}"
        )

    return image, data


def grids_match(
    reference_image: nib.spatialimages.SpatialImage,
    current_image: nib.spatialimages.SpatialImage,
    affine_tolerance: float,
) -> bool:
    """Check whether two images share dimensions and affine."""
    return (
        tuple(reference_image.shape[:3])
        == tuple(current_image.shape[:3])
        and np.allclose(
            reference_image.affine,
            current_image.affine,
            atol=affine_tolerance,
            rtol=0.0,
        )
    )


def require_same_grid(
    reference_image: nib.spatialimages.SpatialImage,
    current_image: nib.spatialimages.SpatialImage,
    reference_name: str,
    current_name: str,
    affine_tolerance: float,
) -> None:
    """Raise an error when two images do not share the same voxel grid."""
    if grids_match(
        reference_image,
        current_image,
        affine_tolerance,
    ):
        return

    raise ValueError(
        "Image-grid mismatch.\n"
        f"Reference: {reference_name}\n"
        f"Reference shape: {reference_image.shape}\n"
        f"Current: {current_name}\n"
        f"Current shape: {current_image.shape}"
    )


def create_image_like(
    reference_image: nib.spatialimages.SpatialImage,
    data: np.ndarray,
) -> nib.Nifti1Image:
    """Create a float32 NIfTI using the reference image grid."""
    output_data = np.asarray(
        data,
        dtype=np.float32,
    )

    output_header = reference_image.header.copy()
    output_header.set_data_dtype(
        np.float32
    )

    return nib.Nifti1Image(
        output_data,
        reference_image.affine,
        output_header,
    )


def is_valid_nifti(
    image_path: Path,
) -> bool:
    """Return True when an output image exists and can be read."""
    if not image_path.is_file():
        return False

    try:
        image = nib.load(
            str(image_path)
        )

        return (
            len(image.shape) == 3
            or (
                len(image.shape) == 4
                and image.shape[-1] == 1
            )
        )

    except Exception:
        return False


def align_trials(
    policy_table: pd.DataFrame,
    pattern_files: dict[int, Path],
) -> list[tuple[int, float, Path]]:
    """Align finite dual-policy values with pattern-change images."""
    aligned_trials: list[
        tuple[
            int,
            float,
            Path,
        ]
    ] = []

    for row in policy_table.itertuples(
        index=False
    ):
        trial_number = int(
            row.global_trial
        )

        policy_value = float(
            row.delta_policy_dual
        )

        pattern_path = pattern_files.get(
            trial_number
        )

        if pattern_path is None:
            continue

        if not np.isfinite(
            policy_value
        ):
            continue

        aligned_trials.append(
            (
                trial_number,
                policy_value,
                pattern_path,
            )
        )

    return aligned_trials


def fit_voxelwise_regression(
    pattern_stack: np.ndarray,
    policy_values: np.ndarray,
    minimum_valid_observations: int,
    minimum_valid_fraction: float,
) -> tuple[np.ndarray, int, int]:
    """
    Fit a separate linear regression at each voxel.

    Model
    -----
    delta_policy_dual
        = intercept + beta * pattern_change

    Each voxel may use a different subset of trials depending on the
    availability of finite pattern-change values.

    Returns
    -------
    beta_volume:
        Voxel-wise pattern-change slope map.

    number_of_estimated_voxels:
        Number of voxels with a successfully estimated slope.

    voxel_observation_threshold:
        Minimum number of valid observations required at each voxel.
    """
    number_of_trials = int(
        pattern_stack.shape[0]
    )

    spatial_shape = pattern_stack.shape[
        1:
    ]

    voxel_matrix = pattern_stack.reshape(
        number_of_trials,
        -1,
    ).astype(
        np.float64,
        copy=False,
    )

    policy_values = np.asarray(
        policy_values,
        dtype=np.float64,
    )

    if policy_values.shape != (
        number_of_trials,
    ):
        raise ValueError(
            "The number of policy values does not match the number "
            "of pattern-change maps."
        )

    if not np.all(
        np.isfinite(
            policy_values
        )
    ):
        raise ValueError(
            "The aligned delta_policy_dual vector contains NaN or Inf."
        )

    fractional_threshold = int(
        math.ceil(
            minimum_valid_fraction
            * number_of_trials
        )
    )

    voxel_observation_threshold = max(
        minimum_valid_observations,
        fractional_threshold,
    )

    finite_counts = np.sum(
        np.isfinite(
            voxel_matrix
        ),
        axis=0,
    )

    candidate_voxels = np.flatnonzero(
        finite_counts
        >= voxel_observation_threshold
    )

    beta_flat = np.full(
        voxel_matrix.shape[1],
        np.nan,
        dtype=np.float32,
    )

    for voxel_index in candidate_voxels:
        predictor = voxel_matrix[
            :,
            voxel_index,
        ]

        valid_observations = np.isfinite(
            predictor
        )

        predictor_valid = predictor[
            valid_observations
        ]

        policy_valid = policy_values[
            valid_observations
        ]

        if (
            predictor_valid.size
            < voxel_observation_threshold
        ):
            continue

        if (
            np.std(
                predictor_valid,
                ddof=0,
            )
            < EPSILON
        ):
            continue

        design_matrix = np.column_stack(
            (
                predictor_valid,
                np.ones_like(
                    predictor_valid
                ),
            )
        )

        coefficients, _, design_rank, _ = np.linalg.lstsq(
            design_matrix,
            policy_valid,
            rcond=None,
        )

        if design_rank < 2:
            continue

        beta_flat[
            voxel_index
        ] = np.float32(
            coefficients[0]
        )

    beta_volume = beta_flat.reshape(
        spatial_shape
    )

    number_of_estimated_voxels = int(
        np.isfinite(
            beta_flat
        ).sum()
    )

    return (
        beta_volume,
        number_of_estimated_voxels,
        voxel_observation_threshold,
    )


def save_alignment_table(
    alignment_root: Path,
    subject_name: str,
    aligned_trials: Sequence[
        tuple[
            int,
            float,
            Path,
        ]
    ],
    pattern_root: Path,
) -> Path:
    """Save the trials included in one participant-level regression."""
    alignment_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    alignment_path = (
        alignment_root
        / (
            f"{subject_name}_"
            "delta_policy_dual_alignment.tsv"
        )
    )

    alignment_table = pd.DataFrame(
        {
            "participant_id": subject_name,
            "global_trial": [
                trial_number
                for (
                    trial_number,
                    _,
                    _,
                ) in aligned_trials
            ],
            "delta_policy_dual": [
                policy_value
                for (
                    _,
                    policy_value,
                    _,
                ) in aligned_trials
            ],
            "pattern_file": [
                relative_path_text(
                    pattern_path,
                    pattern_root,
                )
                for (
                    _,
                    _,
                    pattern_path,
                ) in aligned_trials
            ],
        }
    )

    alignment_table.to_csv(
        alignment_path,
        sep="\t",
        index=False,
        encoding="utf-8",
    )

    return alignment_path


def run_participant(
    subject_name: str,
    dataframe: pd.DataFrame,
    pattern_directory: Path,
    pattern_root: Path,
    output_root: Path,
    output_extension: str,
    minimum_trials: int,
    minimum_valid_observations: int,
    minimum_valid_fraction: float,
    affine_tolerance: float,
    skip_existing: bool,
    save_alignment_tables: bool,
) -> dict[str, object]:
    """Run the dual-policy regression for one participant."""
    output_path = (
        output_root
        / (
            f"{subject_name}_"
            "regression_patternchange2delta_policy_dual"
            f"{output_extension}"
        )
    )

    if (
        skip_existing
        and is_valid_nifti(
            output_path
        )
    ):
        print(
            f"[Existing] {subject_name}: {output_path}"
        )

        return {
            "participant_id": subject_name,
            "status": "existing",
            "output_file": relative_path_text(
                output_path,
                output_root,
            ),
            "reason": "",
        }

    policy_table = prepare_dual_policy_table(
        dataframe=dataframe,
        subject_name=subject_name,
    )

    pattern_files = list_pattern_change_files(
        pattern_directory
    )

    if not pattern_files:
        raise FileNotFoundError(
            "No pattern_change_trial_*.nii or .nii.gz files were found."
        )

    aligned_trials = align_trials(
        policy_table=policy_table,
        pattern_files=pattern_files,
    )

    number_of_matched_trials = len(
        aligned_trials
    )

    if number_of_matched_trials < minimum_trials:
        raise RuntimeError(
            "Too few aligned trials: "
            f"{number_of_matched_trials} < {minimum_trials}."
        )

    print(
        f"[Processing] {subject_name}: "
        f"{number_of_matched_trials} aligned trials; "
        f"trial range "
        f"{aligned_trials[0][0]}-"
        f"{aligned_trials[-1][0]}."
    )

    pattern_arrays: list[np.ndarray] = []

    reference_image = None
    reference_path = None

    policy_values: list[float] = []

    for (
        trial_number,
        policy_value,
        pattern_path,
    ) in aligned_trials:
        image, data = load_3d_image(
            pattern_path
        )

        if reference_image is None:
            reference_image = image
            reference_path = pattern_path

        else:
            require_same_grid(
                reference_image=reference_image,
                current_image=image,
                reference_name=reference_path.name,
                current_name=pattern_path.name,
                affine_tolerance=affine_tolerance,
            )

        pattern_arrays.append(
            data
        )

        policy_values.append(
            policy_value
        )

    if reference_image is None:
        raise RuntimeError(
            "No aligned pattern-change images were loaded."
        )

    pattern_stack = np.stack(
        pattern_arrays,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    policy_array = np.asarray(
        policy_values,
        dtype=np.float64,
    )

    (
        beta_volume,
        number_of_estimated_voxels,
        voxel_observation_threshold,
    ) = fit_voxelwise_regression(
        pattern_stack=pattern_stack,
        policy_values=policy_array,
        minimum_valid_observations=(
            minimum_valid_observations
        ),
        minimum_valid_fraction=(
            minimum_valid_fraction
        ),
    )

    if number_of_estimated_voxels == 0:
        raise RuntimeError(
            "No voxel satisfied the regression requirements."
        )

    output_image = create_image_like(
        reference_image,
        beta_volume,
    )

    nib.save(
        output_image,
        str(output_path),
    )

    alignment_file = ""

    if save_alignment_tables:
        alignment_path = save_alignment_table(
            alignment_root=(
                output_root
                / "alignment_tables"
            ),
            subject_name=subject_name,
            aligned_trials=aligned_trials,
            pattern_root=pattern_root,
        )

        alignment_file = relative_path_text(
            alignment_path,
            output_root,
        )

    print(
        f"[Saved] {subject_name}: "
        f"{number_of_estimated_voxels} voxels -> "
        f"{output_path}"
    )

    return {
        "participant_id": subject_name,
        "n_csv_valid_rows": len(
            policy_table
        ),
        "n_pattern_files": len(
            pattern_files
        ),
        "n_matched_trials": (
            number_of_matched_trials
        ),
        "voxel_observation_threshold": (
            voxel_observation_threshold
        ),
        "n_estimated_voxels": (
            number_of_estimated_voxels
        ),
        "output_file": relative_path_text(
            output_path,
            output_root,
        ),
        "alignment_file": alignment_file,
        "status": "completed",
        "reason": "",
    }


def write_processing_summary(
    output_root: Path,
    summary_rows: list[dict[str, object]],
) -> Path:
    """Write a TSV summary for all participant-level analyses."""
    summary_path = (
        output_root
        / "processing_summary.tsv"
    )

    columns = [
        "participant_id",
        "n_csv_valid_rows",
        "n_pattern_files",
        "n_matched_trials",
        "voxel_observation_threshold",
        "n_estimated_voxels",
        "output_file",
        "alignment_file",
        "status",
        "reason",
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
                    row.get(
                        column,
                        "",
                    )
                )
                for column in columns
            ]

            file.write(
                "\t".join(values)
                + "\n"
            )

    return summary_path


def main() -> None:
    """Run the dual-policy regression for all selected participants."""
    arguments = parse_arguments()

    pattern_root = (
        arguments.pattern_root
        .expanduser()
        .resolve()
    )

    csv_root = (
        arguments.csv_root
        .expanduser()
        .resolve()
    )

    output_root = (
        arguments.output_root
        .expanduser()
        .resolve()
    )

    if not pattern_root.is_dir():
        raise NotADirectoryError(
            f"Pattern-change root does not exist: {pattern_root}"
        )

    if not csv_root.is_dir():
        raise NotADirectoryError(
            f"Trial-wise CSV root does not exist: {csv_root}"
        )

    if arguments.minimum_trials < 3:
        raise ValueError(
            "--minimum-trials must be at least 3."
        )

    if arguments.minimum_valid_observations < 3:
        raise ValueError(
            "--minimum-valid-observations must be at least 3."
        )

    if not (
        0
        < arguments.minimum_valid_fraction
        <= 1
    ):
        raise ValueError(
            "--minimum-valid-fraction must be greater than 0 "
            "and no greater than 1."
        )

    if arguments.affine_tolerance < 0:
        raise ValueError(
            "--affine-tolerance must be zero or greater."
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    participant_records = discover_trialwise_csv_files(
        csv_root=csv_root,
        requested_subject_ids=(
            arguments.subject_ids
        ),
    )

    if not participant_records:
        raise RuntimeError(
            "No matching participant trial-wise CSV files were found."
        )

    print(
        f"Pattern-change root:      {pattern_root}"
    )
    print(
        f"Trial-wise CSV root:      {csv_root}"
    )
    print(
        f"Output root:              {output_root}"
    )
    print(
        f"Participants:             {len(participant_records)}"
    )
    print(
        f"Target:                   {TARGET_COLUMN}"
    )
    print(
        f"Minimum matched trials:   {arguments.minimum_trials}"
    )
    print(
        f"Minimum voxel fraction:   "
        f"{arguments.minimum_valid_fraction:.3f}"
    )
    print(
        f"Minimum voxel samples:    "
        f"{arguments.minimum_valid_observations}"
    )

    summary_rows: list[
        dict[str, object]
    ] = []

    for (
        subject_name,
        csv_path,
    ) in participant_records:
        print(
            f"\n{'=' * 72}\n"
            f"Participant: {subject_name}\n"
            f"{'=' * 72}"
        )

        pattern_directory = find_pattern_directory(
            pattern_root,
            subject_name,
        )

        if pattern_directory is None:
            reason = (
                "Participant pattern-change directory was not found."
            )

            print(
                f"[Skipped] {subject_name}: {reason}"
            )

            summary_rows.append(
                {
                    "participant_id": subject_name,
                    "status": "skipped",
                    "reason": reason,
                }
            )

            continue

        try:
            dataframe = read_csv_with_fallback(
                csv_path
            )

            summary = run_participant(
                subject_name=subject_name,
                dataframe=dataframe,
                pattern_directory=pattern_directory,
                pattern_root=pattern_root,
                output_root=output_root,
                output_extension=(
                    arguments.output_extension
                ),
                minimum_trials=(
                    arguments.minimum_trials
                ),
                minimum_valid_observations=(
                    arguments.minimum_valid_observations
                ),
                minimum_valid_fraction=(
                    arguments.minimum_valid_fraction
                ),
                affine_tolerance=(
                    arguments.affine_tolerance
                ),
                skip_existing=(
                    arguments.skip_existing
                ),
                save_alignment_tables=(
                    arguments.save_alignment_tables
                ),
            )

            summary_rows.append(
                summary
            )

        except Exception as error:
            print(
                f"[Failed] {subject_name}: {error}"
            )

            summary_rows.append(
                {
                    "participant_id": subject_name,
                    "status": "failed",
                    "reason": str(error),
                }
            )

    summary_path = write_processing_summary(
        output_root,
        summary_rows,
    )

    number_completed = sum(
        row.get("status") == "completed"
        for row in summary_rows
    )

    number_existing = sum(
        row.get("status") == "existing"
        for row in summary_rows
    )

    number_failed = sum(
        row.get("status") == "failed"
        for row in summary_rows
    )

    number_skipped = sum(
        row.get("status") == "skipped"
        for row in summary_rows
    )

    print("\n" + "=" * 72)
    print("Analysis completed")
    print("=" * 72)

    print(
        f"Completed analyses: {number_completed}"
    )
    print(
        f"Existing outputs:   {number_existing}"
    )
    print(
        f"Failed analyses:    {number_failed}"
    )
    print(
        f"Skipped analyses:   {number_skipped}"
    )
    print(
        f"Processing summary: {summary_path}"
    )
    print(
        f"Output root:        {output_root}"
    )


if __name__ == "__main__":
    main()