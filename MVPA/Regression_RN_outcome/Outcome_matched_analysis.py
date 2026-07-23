#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
First-level voxel-wise regression for outcome-matched negative vPE trials.

This analysis is restricted to the RN stage and to trials satisfying:

    actual outcome = no reward
    vPE_dual < 0

For each participant and ROI, the voxel-wise model is:

    pattern_change
        ~ seed
        + absPE
        + seed × absPE
        + whole-GM beta
        + intercept

where:

- seed is the trial-wise mean outcome-locked LSS beta within the ROI;
- absPE is abs_vPE_dual from the dual-learning-rate model;
- whole-GM beta is the trial-wise mean outcome-locked LSS beta within
  the whole-gray-matter mask;
- pattern_change is the trial-wise voxel map used as the dependent variable.

All continuous regressors are z-scored within participant across the
included no-reward trials. No temporal detrending is applied.

Important
---------
The whole-GM nuisance regressor is extracted from the same outcome-locked
LSS beta maps used to extract the ROI seed series. It is not calculated
from the pattern-change maps.

Expected directory structure
----------------------------
project/
├── run_rn_no_reward_dual_regression.py
├── data/
│   ├── rois/
│   │   ├── roi_1.nii
│   │   └── roi_2.nii.gz
│   ├── lss_outcome/
│   │   ├── Sub02_LSS_outcome/
│   │   │   ├── beta_outcome_001.nii
│   │   │   └── ...
│   │   └── ...
│   ├── pattern_change_labeled/
│   │   ├── Sub02/
│   │   │   └── RN/
│   │   │       ├── pattern_change_trial_001.nii
│   │   │       └── ...
│   │   └── ...
│   ├── onsets/
│   │   ├── Sub02/
│   │   │   └── csv_output/
│   │   │       └── cue_filtered_trials.csv
│   │   └── ...
│   ├── qlearning_dual/
│   │   ├── Qlearning_single_dual_trialwise_Sub02.csv
│   │   └── ...
│   └── masks/
│       └── group_gm_mask.nii.gz
└── outputs/
    └── rn_no_reward_dual_regression/

Required dual-model columns
---------------------------
global_trial
block
trial_in_block
vPE_dual
abs_vPE_dual

At least one of the following actual-outcome columns is also required:

reward_raw
    Expected coding: 0 = no reward, 1 = reward

reward_model
    Expected coding: -0.5 = no reward, +0.5 = reward

Dependencies
------------
numpy
pandas
nibabel
nilearn  (optional; used only for new_img_like)

Example
-------
python run_rn_no_reward_dual_regression.py \
    --roi-dir /path/to/rois \
    --lss-root /path/to/lss_outcome \
    --pattern-root /path/to/pattern_change_labeled \
    --onsets-root /path/to/onsets \
    --q-root /path/to/qlearning_dual \
    --gm-mask /path/to/group_gm_mask.nii.gz \
    --output-root /path/to/output
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
import pandas as pd

try:
    from nilearn.image import new_img_like

    USE_NILEARN = True
except ImportError:
    USE_NILEARN = False


# =============================================================================
# Fixed column and filename conventions
# =============================================================================

TARGET_LABEL = "RN"

BETA_PREFIX = "beta_outcome_"
PATTERN_PREFIX = "pattern_change_trial_"

Q_SUBJECT_COLUMN = "subject"
Q_GLOBAL_TRIAL_COLUMN = "global_trial"
Q_BLOCK_COLUMN = "block"
Q_TRIAL_IN_BLOCK_COLUMN = "trial_in_block"
Q_REWARD_RAW_COLUMN = "reward_raw"
Q_REWARD_MODEL_COLUMN = "reward_model"
Q_VPE_COLUMN = "vPE_dual"
Q_ABS_VPE_COLUMN = "abs_vPE_dual"

NIFTI_SUFFIXES = (
    ".nii",
    ".nii.gz",
)

EPSILON = 1e-6


# =============================================================================
# Argument parsing
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse portable path and analysis settings."""
    script_directory = Path(__file__).resolve().parent
    default_data_root = script_directory / "data"

    parser = argparse.ArgumentParser(
        description=(
            "Run RN-stage voxel-wise regression on actual no-reward trials "
            "with negative dual-model vPE values."
        )
    )

    parser.add_argument(
        "--roi-dir",
        type=Path,
        default=default_data_root / "rois",
        help=(
            "Directory containing ROI masks. "
            "Default: <script_dir>/data/rois"
        ),
    )

    parser.add_argument(
        "--lss-root",
        type=Path,
        default=default_data_root / "lss_outcome",
        help=(
            "Root directory containing SubXX_LSS_outcome folders. "
            "Default: <script_dir>/data/lss_outcome"
        ),
    )

    parser.add_argument(
        "--pattern-root",
        type=Path,
        default=default_data_root / "pattern_change_labeled",
        help=(
            "Root directory containing SubXX/RN pattern-change folders. "
            "Default: <script_dir>/data/pattern_change_labeled"
        ),
    )

    parser.add_argument(
        "--onsets-root",
        type=Path,
        default=default_data_root / "onsets",
        help=(
            "Root directory containing cue_filtered_trials.csv files. "
            "Default: <script_dir>/data/onsets"
        ),
    )

    parser.add_argument(
        "--q-root",
        type=Path,
        default=default_data_root / "qlearning_dual",
        help=(
            "Directory containing dual-model trial-wise CSV files. "
            "Default: <script_dir>/data/qlearning_dual"
        ),
    )

    parser.add_argument(
        "--gm-mask",
        type=Path,
        default=default_data_root / "masks" / "group_gm_mask.nii.gz",
        help=(
            "Whole-gray-matter mask used for beta extraction and "
            "voxel-wise regression. "
            "Default: <script_dir>/data/masks/group_gm_mask.nii.gz"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            script_directory
            / "outputs"
            / "rn_no_reward_dual_regression"
        ),
        help=(
            "Output directory for first-level maps and QC files. "
            "Default: "
            "<script_dir>/outputs/rn_no_reward_dual_regression"
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
        "--minimum-trials",
        type=int,
        default=12,
        help=(
            "Minimum number of matched no-reward negative-vPE trials. "
            "Default: 12"
        ),
    )

    parser.add_argument(
        "--minimum-dof",
        type=int,
        default=5,
        help=(
            "Minimum residual degrees of freedom required for the model. "
            "Default: 5"
        ),
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
        "--condition-number-warning",
        type=float,
        default=1e4,
        help=(
            "Print a warning when the design-matrix condition number "
            "exceeds this value. Default: 1e4"
        ),
    )

    parser.add_argument(
        "--output-extension",
        choices=(".nii", ".nii.gz"),
        default=".nii",
        help="Output image extension. Default: .nii",
    )

    parser.add_argument(
        "--save-tmaps",
        action="store_true",
        help="Also save participant-level t-statistic maps.",
    )

    parser.add_argument(
        "--skip-design-csv",
        action="store_true",
        help="Do not save participant-level design tables.",
    )

    parser.add_argument(
        "--residualize-seed-by-gm",
        action="store_true",
        help=(
            "Residualize the seed regressor using the whole-GM beta "
            "regressor before constructing the interaction. By default, "
            "GM is included directly as a covariate without pre-residualizing "
            "the seed."
        ),
    )

    parser.add_argument(
        "--allow-outcome-pe-sign-contradictions",
        action="store_true",
        help=(
            "Do not stop when actual outcomes and vPE signs are inconsistent. "
            "By default, such inconsistencies are treated as alignment errors."
        ),
    )

    return parser.parse_args()


# =============================================================================
# General utilities
# =============================================================================

def natural_sort_key(value: str) -> list[object]:
    """Sort strings naturally, for example Sub2 before Sub10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def normalize_column_name(name: str) -> str:
    """Normalize a column name for case-insensitive matching."""
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(name).strip().lower(),
    )


def find_column(
    dataframe: pd.DataFrame,
    candidates: Sequence[str],
) -> str | None:
    """Find a column using normalized candidate names."""
    normalized_columns = {
        normalize_column_name(column): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        normalized_candidate = normalize_column_name(
            candidate
        )

        if normalized_candidate in normalized_columns:
            return normalized_columns[
                normalized_candidate
            ]

    return None


def normalize_subject_id(value: str) -> str:
    """Extract the numeric part of a participant identifier."""
    match = re.search(
        r"(\d+)",
        str(value),
    )

    if match is None:
        raise ValueError(
            f"Could not parse participant ID from: {value}"
        )

    return match.group(1)


def subject_numeric_key(value: str) -> int:
    """Return the numeric participant identifier."""
    return int(
        normalize_subject_id(
            value
        )
    )


def strip_nifti_suffix(path: Path) -> str:
    """Return a filename without .nii or .nii.gz."""
    filename = path.name

    if filename.lower().endswith(".nii.gz"):
        return filename[:-7]

    if filename.lower().endswith(".nii"):
        return filename[:-4]

    return path.stem


def is_nifti_file(path: Path) -> bool:
    """Return True when a path is a supported NIfTI image."""
    return (
        path.is_file()
        and path.name.lower().endswith(
            NIFTI_SUFFIXES
        )
    )


def relative_path_text(
    path: Path,
    root: Path,
) -> str:
    """Return a portable path relative to a specified root."""
    try:
        return path.resolve().relative_to(
            root.resolve()
        ).as_posix()

    except ValueError:
        return path.name


def clean_text_field(value: object) -> str:
    """Remove tabs and line breaks before writing tabular output."""
    return (
        str(value)
        .replace("\t", " ")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def sanitize_error_message(
    error: Exception,
    replacement_roots: dict[str, Path],
) -> str:
    """Replace machine-specific roots in stored error messages."""
    message = str(error)

    for placeholder, root in replacement_roots.items():
        root_text = str(
            root.resolve()
        )

        message = message.replace(
            root_text,
            f"<{placeholder}>",
        )

        message = message.replace(
            root_text.replace("\\", "/"),
            f"<{placeholder}>",
        )

    return clean_text_field(
        message
    )


# =============================================================================
# NIfTI utilities
# =============================================================================

def load_3d_image(
    path: Path,
) -> tuple[
    nib.spatialimages.SpatialImage,
    np.ndarray,
]:
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
            f"Expected a 3D image, received shape {data.shape}: {path}"
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
    """Raise an error when two images do not share a voxel grid."""
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
    """Create a float32 NIfTI using a reference image grid."""
    output_data = np.asarray(
        data,
        dtype=np.float32,
    )

    if USE_NILEARN:
        try:
            return new_img_like(
                reference_image,
                output_data,
                copy_header=True,
            )

        except TypeError:
            return new_img_like(
                reference_image,
                output_data,
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


def save_flat_map(
    values_valid: np.ndarray,
    valid_mask_flat: np.ndarray,
    volume_shape: tuple[int, int, int],
    reference_image: nib.spatialimages.SpatialImage,
    output_path: Path,
) -> None:
    """Insert valid voxel values into a full-volume map and save it."""
    full_data = np.full(
        valid_mask_flat.shape[0],
        np.nan,
        dtype=np.float32,
    )

    full_data[
        valid_mask_flat
    ] = np.asarray(
        values_valid,
        dtype=np.float32,
    )

    volume_data = full_data.reshape(
        volume_shape
    )

    output_image = create_image_like(
        reference_image,
        volume_data,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    nib.save(
        output_image,
        str(output_path),
    )


def first_existing_nifti(
    base_path: Path,
) -> Path | None:
    """Return an existing .nii or .nii.gz version of a path."""
    if base_path.is_file():
        return base_path

    path_text = str(
        base_path
    )

    if path_text.lower().endswith(".nii"):
        compressed_path = Path(
            path_text + ".gz"
        )

        if compressed_path.is_file():
            return compressed_path

    elif not path_text.lower().endswith(".nii.gz"):
        uncompressed_path = Path(
            path_text + ".nii"
        )

        compressed_path = Path(
            path_text + ".nii.gz"
        )

        if uncompressed_path.is_file():
            return uncompressed_path

        if compressed_path.is_file():
            return compressed_path

    return None


# =============================================================================
# Statistical utilities
# =============================================================================

def zscore_vector(
    values: np.ndarray,
    epsilon: float = EPSILON,
) -> np.ndarray:
    """
    Z-score a one-dimensional vector.

    Temporal detrending is not performed.
    """
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            f"Expected a one-dimensional vector, received {values.shape}."
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "The input vector contains NaN or Inf."
        )

    mean_value = np.mean(
        values
    )

    standard_deviation = np.std(
        values,
        ddof=0,
    )

    if (
        not np.isfinite(standard_deviation)
        or standard_deviation < epsilon
    ):
        raise ValueError(
            "The input vector has zero or near-zero variance."
        )

    return (
        values - mean_value
    ) / standard_deviation


def safe_correlation(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> float:
    """Calculate a correlation when both vectors have valid variance."""
    x_values = np.asarray(
        x_values,
        dtype=np.float64,
    )

    y_values = np.asarray(
        y_values,
        dtype=np.float64,
    )

    finite_values = (
        np.isfinite(x_values)
        & np.isfinite(y_values)
    )

    x_values = x_values[
        finite_values
    ]

    y_values = y_values[
        finite_values
    ]

    if (
        x_values.size < 3
        or y_values.size < 3
        or np.std(x_values, ddof=0) < EPSILON
        or np.std(y_values, ddof=0) < EPSILON
    ):
        return np.nan

    return float(
        np.corrcoef(
            x_values,
            y_values,
        )[0, 1]
    )


# =============================================================================
# Participant and file discovery
# =============================================================================

def detect_subject_directories(
    pattern_root: Path,
) -> list[Path]:
    """Detect participant directories matching Sub<number>."""
    subject_directories = [
        path
        for path in pattern_root.iterdir()
        if (
            path.is_dir()
            and re.fullmatch(
                r"Sub\d+",
                path.name,
                flags=re.IGNORECASE,
            )
        )
    ]

    return sorted(
        subject_directories,
        key=lambda path: natural_sort_key(
            path.name
        ),
    )


def find_subject_directory(
    root: Path,
    subject_id: str,
) -> Path | None:
    """Locate Sub2, Sub02, or Sub002 under a root directory."""
    participant_number = int(
        subject_id
    )

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

        checked_names.add(
            candidate_name
        )

        candidate_path = (
            root
            / candidate_name
        )

        if candidate_path.is_dir():
            return candidate_path

    return None


def find_lss_directory(
    lss_root: Path,
    subject_name: str,
) -> Path | None:
    """Locate a participant outcome-locked LSS directory."""
    subject_id = normalize_subject_id(
        subject_name
    )

    participant_number = int(
        subject_id
    )

    candidate_names = [
        f"{subject_name}_LSS_outcome",
        f"Sub{subject_id}_LSS_outcome",
        f"Sub{participant_number}_LSS_outcome",
        f"Sub{participant_number:02d}_LSS_outcome",
        f"Sub{participant_number:03d}_LSS_outcome",
    ]

    checked_names: set[str] = set()

    for candidate_name in candidate_names:
        if candidate_name in checked_names:
            continue

        checked_names.add(
            candidate_name
        )

        candidate_path = (
            lss_root
            / candidate_name
        )

        if candidate_path.is_dir():
            return candidate_path

    return None


def find_qlearning_file(
    q_root: Path,
    subject_name: str,
) -> Path | None:
    """Locate a participant dual-model trial-wise CSV file."""
    subject_id = normalize_subject_id(
        subject_name
    )

    participant_number = int(
        subject_id
    )

    subject_variants = [
        subject_name,
        subject_name.lower(),
        f"Sub{subject_id}",
        f"Sub{participant_number}",
        f"Sub{participant_number:02d}",
        f"Sub{participant_number:03d}",
    ]

    exact_candidates: list[Path] = []

    for subject_variant in subject_variants:
        exact_candidates.append(
            q_root
            / (
                "Qlearning_single_dual_trialwise_"
                f"{subject_variant}.csv"
            )
        )

    for candidate_path in exact_candidates:
        if candidate_path.is_file():
            return candidate_path

    matches = sorted(
        q_root.glob(
            f"Qlearning_single_dual_trialwise_*{subject_id}*.csv"
        ),
        key=lambda path: natural_sort_key(
            path.name
        ),
    )

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        match_names = ", ".join(
            path.name
            for path in matches
        )

        raise RuntimeError(
            f"Multiple dual-model CSV files matched {subject_name}: "
            f"{match_names}"
        )

    return None


def get_subject_directories(
    pattern_root: Path,
    requested_subject_ids: Sequence[str] | None,
) -> list[Path]:
    """Return selected participant directories or detect all."""
    if requested_subject_ids is None:
        return detect_subject_directories(
            pattern_root
        )

    selected_directories: list[Path] = []

    for requested_id in requested_subject_ids:
        subject_id = normalize_subject_id(
            requested_id
        )

        subject_directory = find_subject_directory(
            pattern_root,
            subject_id,
        )

        if subject_directory is None:
            print(
                f"[Missing] Pattern-change directory not found "
                f"for participant {requested_id}."
            )
            continue

        selected_directories.append(
            subject_directory
        )

    unique_directories = {
        directory.resolve(): directory
        for directory in selected_directories
    }

    return sorted(
        unique_directories.values(),
        key=lambda path: natural_sort_key(
            path.name
        ),
    )


def trial_from_filename(
    filename: str,
    prefix: str,
) -> int | None:
    """Extract the trial number from a prefixed NIfTI filename."""
    if not filename.lower().startswith(
        prefix.lower()
    ):
        return None

    remainder = filename[
        len(prefix):
    ]

    if remainder.lower().endswith(".nii.gz"):
        remainder = remainder[:-7]

    elif remainder.lower().endswith(".nii"):
        remainder = remainder[:-4]

    match = re.search(
        r"(\d+)",
        remainder,
    )

    if match is None:
        return None

    return int(
        match.group(1)
    )


def list_pattern_files(
    pattern_directory: Path,
) -> list[Path]:
    """
    List pattern-change maps sorted by trial number.

    When both .nii and .nii.gz versions of one trial exist, the compressed
    version is preferred to prevent duplicate trials.
    """
    files_by_trial: dict[int, Path] = {}

    if not pattern_directory.is_dir():
        return []

    for image_path in pattern_directory.iterdir():
        if not is_nifti_file(
            image_path
        ):
            continue

        trial_number = trial_from_filename(
            image_path.name,
            PATTERN_PREFIX,
        )

        if trial_number is None:
            continue

        existing_path = files_by_trial.get(
            trial_number
        )

        if existing_path is None:
            files_by_trial[
                trial_number
            ] = image_path
            continue

        if (
            image_path.name.lower().endswith(".nii.gz")
            and not existing_path.name.lower().endswith(".nii.gz")
        ):
            files_by_trial[
                trial_number
            ] = image_path

    return [
        files_by_trial[trial_number]
        for trial_number in sorted(
            files_by_trial
        )
    ]


# =============================================================================
# CSV loading and trial alignment
# =============================================================================

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


def load_rn_onset_table(
    onset_path: Path,
) -> pd.DataFrame:
    """
    Read global trial numbers and stage labels.

    The onset-table trial number must correspond to global_trial in the
    dual-model trial-wise table.
    """
    onset_table = read_csv_with_fallback(
        onset_path
    )

    trial_column = find_column(
        onset_table,
        [
            "trial",
            "global_trial",
            "globaltrial",
        ],
    )

    label_column = find_column(
        onset_table,
        ["label"],
    )

    if (
        trial_column is None
        or label_column is None
    ):
        raise ValueError(
            "The onset CSV lacks a trial/global_trial column or label column."
        )

    onset_table = onset_table[
        [
            trial_column,
            label_column,
        ]
    ].copy()

    onset_table = onset_table.rename(
        columns={
            trial_column: "global_trial",
            label_column: "label",
        }
    )

    onset_table["global_trial"] = pd.to_numeric(
        onset_table["global_trial"],
        errors="coerce",
    )

    onset_table["label"] = (
        onset_table["label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    onset_table = onset_table.dropna(
        subset=[
            "global_trial",
        ]
    ).copy()

    onset_table["global_trial"] = (
        onset_table["global_trial"]
        .astype(int)
    )

    duplicated_trials = onset_table.duplicated(
        subset=[
            "global_trial",
        ],
        keep=False,
    )

    if duplicated_trials.any():
        duplicate_ids = (
            onset_table.loc[
                duplicated_trials,
                "global_trial",
            ]
            .astype(int)
            .tolist()
        )

        raise ValueError(
            "The onset CSV contains duplicated global_trial values: "
            f"{duplicate_ids[:20]}"
        )

    return onset_table.loc[
        onset_table["label"] == TARGET_LABEL
    ].copy()


def validate_subject_column(
    q_table: pd.DataFrame,
    expected_subject: str,
) -> None:
    """Verify the optional subject column against the folder participant."""
    if Q_SUBJECT_COLUMN not in q_table.columns:
        return

    observed_values = (
        q_table[
            Q_SUBJECT_COLUMN
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    if not observed_values:
        return

    expected_number = subject_numeric_key(
        expected_subject
    )

    observed_numbers: list[int] = []

    for observed_value in observed_values:
        try:
            observed_numbers.append(
                subject_numeric_key(
                    observed_value
                )
            )

        except ValueError:
            raise ValueError(
                "The Q-learning subject column contains an invalid "
                f"participant identifier: {observed_value}"
            )

    if (
        len(set(observed_numbers)) != 1
        or observed_numbers[0] != expected_number
    ):
        raise ValueError(
            "The participant recorded in the Q-learning table does not "
            f"match {expected_subject}: {observed_values}"
        )


def load_dual_qlearning_table(
    q_path: Path,
    expected_subject: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read and validate a dual-learning-rate trial-wise table."""
    q_table = read_csv_with_fallback(
        q_path
    )

    required_columns = [
        Q_GLOBAL_TRIAL_COLUMN,
        Q_BLOCK_COLUMN,
        Q_TRIAL_IN_BLOCK_COLUMN,
        Q_VPE_COLUMN,
        Q_ABS_VPE_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in q_table.columns
    ]

    if missing_columns:
        raise ValueError(
            "The dual-model CSV lacks required columns: "
            f"{missing_columns}"
        )

    validate_subject_column(
        q_table,
        expected_subject,
    )

    numeric_columns = [
        Q_GLOBAL_TRIAL_COLUMN,
        Q_BLOCK_COLUMN,
        Q_TRIAL_IN_BLOCK_COLUMN,
        Q_VPE_COLUMN,
        Q_ABS_VPE_COLUMN,
    ]

    for column in numeric_columns:
        q_table[column] = pd.to_numeric(
            q_table[column],
            errors="coerce",
        )

    q_table = q_table.dropna(
        subset=numeric_columns
    ).copy()

    if q_table.empty:
        raise ValueError(
            "No valid rows remained in the dual-model table."
        )

    q_table[
        Q_GLOBAL_TRIAL_COLUMN
    ] = q_table[
        Q_GLOBAL_TRIAL_COLUMN
    ].astype(int)

    q_table[
        Q_BLOCK_COLUMN
    ] = q_table[
        Q_BLOCK_COLUMN
    ].astype(int)

    q_table[
        Q_TRIAL_IN_BLOCK_COLUMN
    ] = q_table[
        Q_TRIAL_IN_BLOCK_COLUMN
    ].astype(int)

    duplicated_trials = q_table.duplicated(
        subset=[
            Q_GLOBAL_TRIAL_COLUMN,
        ],
        keep=False,
    )

    if duplicated_trials.any():
        duplicate_ids = (
            q_table.loc[
                duplicated_trials,
                Q_GLOBAL_TRIAL_COLUMN,
            ]
            .astype(int)
            .tolist()
        )

        raise ValueError(
            "The dual-model CSV contains duplicated global_trial values: "
            f"{duplicate_ids[:20]}"
        )

    abs_difference = np.abs(
        q_table[
            Q_ABS_VPE_COLUMN
        ].to_numpy(
            dtype=float
        )
        - np.abs(
            q_table[
                Q_VPE_COLUMN
            ].to_numpy(
                dtype=float
            )
        )
    )

    maximum_abs_difference = float(
        np.max(
            abs_difference
        )
    )

    if maximum_abs_difference > 1e-6:
        raise ValueError(
            "abs_vPE_dual is inconsistent with abs(vPE_dual). "
            f"Maximum difference: {maximum_abs_difference:.8g}"
        )

    reward_source = None

    if Q_REWARD_RAW_COLUMN in q_table.columns:
        q_table[
            Q_REWARD_RAW_COLUMN
        ] = pd.to_numeric(
            q_table[
                Q_REWARD_RAW_COLUMN
            ],
            errors="coerce",
        )

        if q_table[
            Q_REWARD_RAW_COLUMN
        ].isna().any():
            raise ValueError(
                "reward_raw contains values that cannot be parsed."
            )

        reward_values = set(
            np.unique(
                q_table[
                    Q_REWARD_RAW_COLUMN
                ].to_numpy(
                    dtype=float
                )
            ).tolist()
        )

        if not reward_values.issubset(
            {
                0.0,
                1.0,
            }
        ):
            raise ValueError(
                "reward_raw must use 0/1 coding. "
                f"Observed values: {sorted(reward_values)}"
            )

        q_table["no_reward"] = (
            q_table[
                Q_REWARD_RAW_COLUMN
            ] == 0
        ).astype(int)

        reward_source = (
            Q_REWARD_RAW_COLUMN
        )

    elif Q_REWARD_MODEL_COLUMN in q_table.columns:
        q_table[
            Q_REWARD_MODEL_COLUMN
        ] = pd.to_numeric(
            q_table[
                Q_REWARD_MODEL_COLUMN
            ],
            errors="coerce",
        )

        if q_table[
            Q_REWARD_MODEL_COLUMN
        ].isna().any():
            raise ValueError(
                "reward_model contains values that cannot be parsed."
            )

        reward_values = set(
            np.round(
                np.unique(
                    q_table[
                        Q_REWARD_MODEL_COLUMN
                    ].to_numpy(
                        dtype=float
                    )
                ),
                6,
            ).tolist()
        )

        if not reward_values.issubset(
            {
                -0.5,
                0.5,
            }
        ):
            raise ValueError(
                "reward_model must use -0.5/+0.5 coding. "
                f"Observed values: {sorted(reward_values)}"
            )

        q_table["no_reward"] = (
            q_table[
                Q_REWARD_MODEL_COLUMN
            ] < 0
        ).astype(int)

        reward_source = (
            Q_REWARD_MODEL_COLUMN
        )

    else:
        raise ValueError(
            "The dual-model CSV does not contain an actual-outcome column. "
            "At least reward_raw or reward_model is required."
        )

    reward_coding_mismatches = 0

    if (
        Q_REWARD_RAW_COLUMN in q_table.columns
        and Q_REWARD_MODEL_COLUMN in q_table.columns
    ):
        q_table[
            Q_REWARD_MODEL_COLUMN
        ] = pd.to_numeric(
            q_table[
                Q_REWARD_MODEL_COLUMN
            ],
            errors="coerce",
        )

        if q_table[
            Q_REWARD_MODEL_COLUMN
        ].isna().any():
            raise ValueError(
                "reward_model contains values that cannot be parsed."
            )

        expected_model_reward = (
            q_table[
                Q_REWARD_RAW_COLUMN
            ].to_numpy(
                dtype=float
            )
            - 0.5
        )

        observed_model_reward = (
            q_table[
                Q_REWARD_MODEL_COLUMN
            ].to_numpy(
                dtype=float
            )
        )

        reward_coding_mismatches = int(
            np.sum(
                ~np.isclose(
                    expected_model_reward,
                    observed_model_reward,
                    atol=1e-6,
                    rtol=0.0,
                )
            )
        )

        if reward_coding_mismatches > 0:
            raise ValueError(
                "reward_raw and reward_model are inconsistent. "
                f"Mismatched rows: {reward_coding_mismatches}"
            )

    columns_to_keep = [
        Q_GLOBAL_TRIAL_COLUMN,
        Q_BLOCK_COLUMN,
        Q_TRIAL_IN_BLOCK_COLUMN,
        Q_VPE_COLUMN,
        Q_ABS_VPE_COLUMN,
        "no_reward",
    ]

    for optional_column in (
        Q_SUBJECT_COLUMN,
        Q_REWARD_RAW_COLUMN,
        Q_REWARD_MODEL_COLUMN,
        "vPE_sign_dual",
        "winner_by_bic",
    ):
        if optional_column in q_table.columns:
            columns_to_keep.append(
                optional_column
            )

    q_table = q_table[
        list(
            dict.fromkeys(
                columns_to_keep
            )
        )
    ].copy()

    metadata = {
        "reward_source": reward_source,
        "max_abs_vpe_check_difference": maximum_abs_difference,
        "reward_coding_mismatch": reward_coding_mismatches,
        "n_q_rows": int(
            len(q_table)
        ),
    }

    return q_table, metadata


def build_rn_trial_table(
    onset_rn: pd.DataFrame,
    q_table: pd.DataFrame,
    fail_on_sign_contradiction: bool,
) -> pd.DataFrame:
    """Merge RN stage labels with the dual-model table by global trial."""
    merged_table = onset_rn.merge(
        q_table,
        on=Q_GLOBAL_TRIAL_COLUMN,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing_q_rows = int(
        (
            merged_table["_merge"]
            != "both"
        ).sum()
    )

    if missing_q_rows > 0:
        missing_trials = (
            merged_table.loc[
                merged_table["_merge"] != "both",
                Q_GLOBAL_TRIAL_COLUMN,
            ]
            .astype(int)
            .tolist()
        )

        raise ValueError(
            "Some RN trials have no dual-model record. "
            f"Missing trials: {missing_trials[:30]}"
        )

    merged_table = merged_table.drop(
        columns=[
            "_merge",
        ]
    )

    no_reward_positive_vpe = int(
        (
            (merged_table["no_reward"] == 1)
            & (
                merged_table[
                    Q_VPE_COLUMN
                ] > EPSILON
            )
        ).sum()
    )

    reward_negative_vpe = int(
        (
            (merged_table["no_reward"] == 0)
            & (
                merged_table[
                    Q_VPE_COLUMN
                ] < -EPSILON
            )
        ).sum()
    )

    no_reward_zero_vpe = int(
        (
            (merged_table["no_reward"] == 1)
            & (
                np.abs(
                    merged_table[
                        Q_VPE_COLUMN
                    ]
                )
                <= EPSILON
            )
        ).sum()
    )

    reward_zero_vpe = int(
        (
            (merged_table["no_reward"] == 0)
            & (
                np.abs(
                    merged_table[
                        Q_VPE_COLUMN
                    ]
                )
                <= EPSILON
            )
        ).sum()
    )

    if (
        fail_on_sign_contradiction
        and (
            no_reward_positive_vpe > 0
            or reward_negative_vpe > 0
        )
    ):
        raise ValueError(
            "Actual outcomes and dual-model vPE signs are inconsistent. "
            f"No-reward trials with vPE > 0: {no_reward_positive_vpe}; "
            f"reward trials with vPE < 0: {reward_negative_vpe}."
        )

    merged_table.attrs[
        "no_reward_positive_vpe"
    ] = no_reward_positive_vpe

    merged_table.attrs[
        "reward_negative_vpe"
    ] = reward_negative_vpe

    merged_table.attrs[
        "no_reward_zero_vpe"
    ] = no_reward_zero_vpe

    merged_table.attrs[
        "reward_zero_vpe"
    ] = reward_zero_vpe

    return merged_table


# =============================================================================
# Mask loading
# =============================================================================

def load_roi_masks(
    roi_directory: Path,
) -> dict[
    str,
    tuple[
        nib.spatialimages.SpatialImage,
        np.ndarray,
    ],
]:
    """Load all non-empty ROI masks."""
    roi_files = sorted(
        [
            path
            for path in roi_directory.iterdir()
            if is_nifti_file(
                path
            )
        ],
        key=lambda path: natural_sort_key(
            path.name
        ),
    )

    if not roi_files:
        raise FileNotFoundError(
            "No ROI NIfTI files were found."
        )

    roi_masks = {}

    for roi_path in roi_files:
        roi_image, roi_data = load_3d_image(
            roi_path
        )

        roi_mask = (
            np.isfinite(
                roi_data
            )
            & (
                roi_data > 0
            )
        )

        if not np.any(
            roi_mask
        ):
            print(
                f"[Warning] Empty ROI mask skipped: {roi_path.name}"
            )
            continue

        roi_name = strip_nifti_suffix(
            roi_path
        )

        if roi_name in roi_masks:
            raise ValueError(
                f"Duplicated ROI name after suffix removal: {roi_name}"
            )

        roi_masks[
            roi_name
        ] = (
            roi_image,
            roi_mask,
        )

    if not roi_masks:
        raise RuntimeError(
            "All ROI masks were empty or invalid."
        )

    return roi_masks


# =============================================================================
# Main analysis
# =============================================================================

def main() -> None:
    """Run all participant-by-ROI first-level models."""
    arguments = parse_arguments()

    roi_directory = (
        arguments.roi_dir
        .expanduser()
        .resolve()
    )

    lss_root = (
        arguments.lss_root
        .expanduser()
        .resolve()
    )

    pattern_root = (
        arguments.pattern_root
        .expanduser()
        .resolve()
    )

    onsets_root = (
        arguments.onsets_root
        .expanduser()
        .resolve()
    )

    q_root = (
        arguments.q_root
        .expanduser()
        .resolve()
    )

    gm_mask_path = (
        arguments.gm_mask
        .expanduser()
        .resolve()
    )

    output_root = (
        arguments.output_root
        .expanduser()
        .resolve()
    )

    required_directories = {
        "ROI directory": roi_directory,
        "LSS root": lss_root,
        "Pattern-change root": pattern_root,
        "Onsets root": onsets_root,
        "Dual-model root": q_root,
    }

    for description, directory in required_directories.items():
        if not directory.is_dir():
            raise NotADirectoryError(
                f"{description} does not exist: {directory}"
            )

    if not gm_mask_path.is_file():
        raise FileNotFoundError(
            f"Whole-GM mask does not exist: {gm_mask_path}"
        )

    if arguments.minimum_trials < 5:
        raise ValueError(
            "--minimum-trials must be at least 5 because the model "
            "contains five parameters."
        )

    if arguments.minimum_dof < 1:
        raise ValueError(
            "--minimum-dof must be at least 1."
        )

    if arguments.affine_tolerance < 0:
        raise ValueError(
            "--affine-tolerance must be zero or greater."
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    roi_masks = load_roi_masks(
        roi_directory
    )

    gm_image, gm_data = load_3d_image(
        gm_mask_path
    )

    gm_mask = (
        np.isfinite(
            gm_data
        )
        & (
            gm_data > 0
        )
    )

    if not np.any(
        gm_mask
    ):
        raise ValueError(
            "The whole-GM mask contains no positive voxels."
        )

    subject_directories = get_subject_directories(
        pattern_root,
        arguments.subject_ids,
    )

    if not subject_directories:
        raise RuntimeError(
            "No participant pattern-change directories were found."
        )

    replacement_roots = {
        "ROI_ROOT": roi_directory,
        "LSS_ROOT": lss_root,
        "PATTERN_ROOT": pattern_root,
        "ONSETS_ROOT": onsets_root,
        "Q_ROOT": q_root,
        "GM_MASK_ROOT": gm_mask_path.parent,
        "OUTPUT_ROOT": output_root,
    }

    print(
        f"ROI directory:             {roi_directory}"
    )
    print(
        f"LSS root:                  {lss_root}"
    )
    print(
        f"Pattern-change root:       {pattern_root}"
    )
    print(
        f"Onsets root:               {onsets_root}"
    )
    print(
        f"Dual-model root:           {q_root}"
    )
    print(
        f"Whole-GM mask:             {gm_mask_path}"
    )
    print(
        f"Output root:               {output_root}"
    )
    print(
        f"ROIs:                      {len(roi_masks)}"
    )
    print(
        f"Participants:              {len(subject_directories)}"
    )
    print(
        f"Minimum matched trials:    {arguments.minimum_trials}"
    )
    print(
        f"Minimum residual DOF:      {arguments.minimum_dof}"
    )
    print(
        f"Save t-maps:               {arguments.save_tmaps}"
    )
    print(
        f"Save design CSV:           {not arguments.skip_design_csv}"
    )
    print(
        f"Residualize seed by GM:    "
        f"{arguments.residualize_seed_by_gm}"
    )

    qc_rows: list[
        dict[str, Any]
    ] = []

    map_manifest_rows: list[
        dict[str, Any]
    ] = []

    for roi_name, (
        roi_image,
        roi_mask,
    ) in roi_masks.items():
        print(
            f"\n{'=' * 80}\n"
            f"ROI: {roi_name}\n"
            f"{'=' * 80}"
        )

        roi_output_directory = (
            output_root
            / roi_name
        )

        roi_output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for subject_directory in subject_directories:
            subject_name = (
                subject_directory.name
            )

            subject_id = normalize_subject_id(
                subject_name
            )

            lss_directory = find_lss_directory(
                lss_root,
                subject_name,
            )

            onset_subject_directory = find_subject_directory(
                onsets_root,
                subject_id,
            )

            q_path = find_qlearning_file(
                q_root,
                subject_name,
            )

            base_qc: dict[str, Any] = {
                "subject": subject_name,
                "roi": roi_name,
                "label": TARGET_LABEL,
                "status": "not_started",
                "q_file": (
                    relative_path_text(
                        q_path,
                        q_root,
                    )
                    if q_path is not None
                    else ""
                ),
                "outcome_source": "",
                "n_q_rows": np.nan,
                "n_rn_label_trials": np.nan,
                "n_rn_merged_trials": np.nan,
                "n_pattern_files_rn": np.nan,
                "n_valid_pairs_all_rn": np.nan,
                "n_no_reward_pairs": np.nan,
                "n_reward_pairs": np.nan,
                "n_no_reward_negative_pe_pairs": np.nan,
                "n_no_reward_zero_pe_pairs": np.nan,
                "n_no_reward_positive_pe_pairs": np.nan,
                "n_reward_negative_pe_pairs": np.nan,
                "n_reward_zero_pe_pairs": np.nan,
                "n_used_in_model": np.nan,
                "min_required_trials": (
                    arguments.minimum_trials
                ),
                "rank": np.nan,
                "dof": np.nan,
                "condition_number_X": np.nan,
                "mean_absPE": np.nan,
                "std_absPE": np.nan,
                "min_absPE": np.nan,
                "max_absPE": np.nan,
                "mean_seed": np.nan,
                "std_seed": np.nan,
                "mean_GM_beta": np.nan,
                "std_GM_beta": np.nan,
                "corr_seed_absPE": np.nan,
                "corr_absPE_GM": np.nan,
                "corr_seed_GM": np.nan,
                "max_abs_vpe_check_difference": np.nan,
                "reward_coding_mismatch": np.nan,
                "error_message": "",
            }

            try:
                if lss_directory is None:
                    raise FileNotFoundError(
                        "The participant LSS directory was not found."
                    )

                if onset_subject_directory is None:
                    raise FileNotFoundError(
                        "The participant onset directory was not found."
                    )

                if q_path is None:
                    raise FileNotFoundError(
                        "The participant dual-model CSV was not found."
                    )

                onset_path = (
                    onset_subject_directory
                    / "csv_output"
                    / "cue_filtered_trials.csv"
                )

                if not onset_path.is_file():
                    raise FileNotFoundError(
                        "cue_filtered_trials.csv was not found."
                    )

                rn_pattern_directory = (
                    subject_directory
                    / TARGET_LABEL
                )

                if not rn_pattern_directory.is_dir():
                    raise FileNotFoundError(
                        "The RN pattern-change directory was not found."
                    )

                onset_rn = load_rn_onset_table(
                    onset_path
                )

                if onset_rn.empty:
                    raise ValueError(
                        "No RN trials were found in the onset table."
                    )

                (
                    q_table,
                    q_metadata,
                ) = load_dual_qlearning_table(
                    q_path=q_path,
                    expected_subject=subject_name,
                )

                rn_table = build_rn_trial_table(
                    onset_rn=onset_rn,
                    q_table=q_table,
                    fail_on_sign_contradiction=(
                        not arguments
                        .allow_outcome_pe_sign_contradictions
                    ),
                )

                base_qc[
                    "outcome_source"
                ] = q_metadata[
                    "reward_source"
                ]

                base_qc[
                    "n_q_rows"
                ] = q_metadata[
                    "n_q_rows"
                ]

                base_qc[
                    "max_abs_vpe_check_difference"
                ] = q_metadata[
                    "max_abs_vpe_check_difference"
                ]

                base_qc[
                    "reward_coding_mismatch"
                ] = q_metadata[
                    "reward_coding_mismatch"
                ]

                base_qc[
                    "n_rn_label_trials"
                ] = int(
                    len(
                        onset_rn
                    )
                )

                base_qc[
                    "n_rn_merged_trials"
                ] = int(
                    len(
                        rn_table
                    )
                )

                base_qc[
                    "n_no_reward_zero_pe_pairs"
                ] = rn_table.attrs[
                    "no_reward_zero_vpe"
                ]

                base_qc[
                    "n_no_reward_positive_pe_pairs"
                ] = rn_table.attrs[
                    "no_reward_positive_vpe"
                ]

                base_qc[
                    "n_reward_negative_pe_pairs"
                ] = rn_table.attrs[
                    "reward_negative_vpe"
                ]

                base_qc[
                    "n_reward_zero_pe_pairs"
                ] = rn_table.attrs[
                    "reward_zero_vpe"
                ]

                trial_lookup = (
                    rn_table
                    .set_index(
                        Q_GLOBAL_TRIAL_COLUMN
                    )
                    .to_dict(
                        orient="index"
                    )
                )

                pattern_files = list_pattern_files(
                    rn_pattern_directory
                )

                base_qc[
                    "n_pattern_files_rn"
                ] = int(
                    len(
                        pattern_files
                    )
                )

                if not pattern_files:
                    raise FileNotFoundError(
                        "No RN pattern-change maps were found."
                    )

                all_pairs: list[
                    dict[str, Any]
                ] = []

                for pattern_path in pattern_files:
                    global_trial = trial_from_filename(
                        pattern_path.name,
                        PATTERN_PREFIX,
                    )

                    if global_trial is None:
                        continue

                    if global_trial not in trial_lookup:
                        continue

                    beta_path = first_existing_nifti(
                        lss_directory
                        / (
                            f"{BETA_PREFIX}"
                            f"{global_trial:03d}.nii"
                        )
                    )

                    if beta_path is None:
                        print(
                            f"[Missing] {subject_name}, "
                            f"trial {global_trial:03d}: "
                            "outcome beta not found."
                        )
                        continue

                    trial_row = trial_lookup[
                        global_trial
                    ]

                    all_pairs.append(
                        {
                            "global_trial": int(
                                global_trial
                            ),
                            "block": int(
                                trial_row[
                                    Q_BLOCK_COLUMN
                                ]
                            ),
                            "trial_in_block": int(
                                trial_row[
                                    Q_TRIAL_IN_BLOCK_COLUMN
                                ]
                            ),
                            "pattern_path": pattern_path,
                            "beta_path": beta_path,
                            "vPE_dual": float(
                                trial_row[
                                    Q_VPE_COLUMN
                                ]
                            ),
                            "abs_vPE_dual": float(
                                trial_row[
                                    Q_ABS_VPE_COLUMN
                                ]
                            ),
                            "no_reward": int(
                                trial_row[
                                    "no_reward"
                                ]
                            ),
                            "reward_raw": (
                                float(
                                    trial_row[
                                        Q_REWARD_RAW_COLUMN
                                    ]
                                )
                                if (
                                    Q_REWARD_RAW_COLUMN
                                    in trial_row
                                )
                                else np.nan
                            ),
                            "reward_model": (
                                float(
                                    trial_row[
                                        Q_REWARD_MODEL_COLUMN
                                    ]
                                )
                                if (
                                    Q_REWARD_MODEL_COLUMN
                                    in trial_row
                                )
                                else np.nan
                            ),
                        }
                    )

                if not all_pairs:
                    raise RuntimeError(
                        "No RN trials had both a pattern-change map "
                        "and an outcome beta map."
                    )

                all_pairs = sorted(
                    all_pairs,
                    key=lambda row: row[
                        "global_trial"
                    ],
                )

                base_qc[
                    "n_valid_pairs_all_rn"
                ] = int(
                    len(
                        all_pairs
                    )
                )

                base_qc[
                    "n_no_reward_pairs"
                ] = int(
                    sum(
                        row["no_reward"] == 1
                        for row in all_pairs
                    )
                )

                base_qc[
                    "n_reward_pairs"
                ] = int(
                    sum(
                        row["no_reward"] == 0
                        for row in all_pairs
                    )
                )

                base_qc[
                    "n_no_reward_negative_pe_pairs"
                ] = int(
                    sum(
                        (
                            row["no_reward"] == 1
                            and row["vPE_dual"] < -EPSILON
                        )
                        for row in all_pairs
                    )
                )

                used_pairs = [
                    row
                    for row in all_pairs
                    if (
                        row["no_reward"] == 1
                        and row["vPE_dual"] < -EPSILON
                    )
                ]

                if (
                    len(
                        used_pairs
                    )
                    < arguments.minimum_trials
                ):
                    raise RuntimeError(
                        "Too few actual no-reward negative-vPE trials: "
                        f"{len(used_pairs)} < "
                        f"{arguments.minimum_trials}"
                    )

                base_qc[
                    "n_used_in_model"
                ] = int(
                    len(
                        used_pairs
                    )
                )

                seed_values: list[float] = []
                gm_beta_values: list[float] = []

                pattern_arrays: list[
                    np.ndarray
                ] = []

                pattern_reference_image = None
                pattern_reference_path = None

                for pair in used_pairs:
                    beta_image, beta_data = load_3d_image(
                        pair["beta_path"]
                    )

                    require_same_grid(
                        reference_image=roi_image,
                        current_image=beta_image,
                        reference_name=f"ROI {roi_name}",
                        current_name=pair[
                            "beta_path"
                        ].name,
                        affine_tolerance=(
                            arguments.affine_tolerance
                        ),
                    )

                    require_same_grid(
                        reference_image=gm_image,
                        current_image=beta_image,
                        reference_name="whole-GM mask",
                        current_name=pair[
                            "beta_path"
                        ].name,
                        affine_tolerance=(
                            arguments.affine_tolerance
                        ),
                    )

                    roi_beta_values = beta_data[
                        roi_mask
                    ]

                    if not np.any(
                        np.isfinite(
                            roi_beta_values
                        )
                    ):
                        raise ValueError(
                            "An LSS beta map contains no finite "
                            "values within the ROI."
                        )

                    gm_beta_trial_values = beta_data[
                        gm_mask
                    ]

                    if not np.any(
                        np.isfinite(
                            gm_beta_trial_values
                        )
                    ):
                        raise ValueError(
                            "An LSS beta map contains no finite "
                            "values within the whole-GM mask."
                        )

                    seed_value = float(
                        np.nanmean(
                            roi_beta_values
                        )
                    )

                    gm_beta_value = float(
                        np.nanmean(
                            gm_beta_trial_values
                        )
                    )

                    if (
                        not np.isfinite(
                            seed_value
                        )
                        or not np.isfinite(
                            gm_beta_value
                        )
                    ):
                        raise ValueError(
                            "A trial-wise ROI or whole-GM beta value "
                            "is not finite."
                        )

                    pattern_image, pattern_data = load_3d_image(
                        pair[
                            "pattern_path"
                        ]
                    )

                    require_same_grid(
                        reference_image=beta_image,
                        current_image=pattern_image,
                        reference_name=pair[
                            "beta_path"
                        ].name,
                        current_name=pair[
                            "pattern_path"
                        ].name,
                        affine_tolerance=(
                            arguments.affine_tolerance
                        ),
                    )

                    if pattern_reference_image is None:
                        pattern_reference_image = (
                            pattern_image
                        )

                        pattern_reference_path = pair[
                            "pattern_path"
                        ]

                    else:
                        require_same_grid(
                            reference_image=(
                                pattern_reference_image
                            ),
                            current_image=pattern_image,
                            reference_name=(
                                pattern_reference_path.name
                            ),
                            current_name=pair[
                                "pattern_path"
                            ].name,
                            affine_tolerance=(
                                arguments.affine_tolerance
                            ),
                        )

                    seed_values.append(
                        seed_value
                    )

                    gm_beta_values.append(
                        gm_beta_value
                    )

                    pattern_arrays.append(
                        pattern_data
                    )

                seed_array = np.asarray(
                    seed_values,
                    dtype=np.float64,
                )

                gm_beta_array = np.asarray(
                    gm_beta_values,
                    dtype=np.float64,
                )

                pattern_stack = np.stack(
                    pattern_arrays,
                    axis=0,
                ).astype(
                    np.float32,
                    copy=False,
                )

                volume_shape = tuple(
                    int(value)
                    for value in pattern_stack.shape[
                        1:
                    ]
                )

                vpe_values = np.asarray(
                    [
                        row["vPE_dual"]
                        for row in used_pairs
                    ],
                    dtype=np.float64,
                )

                abs_pe_values = np.asarray(
                    [
                        row["abs_vPE_dual"]
                        for row in used_pairs
                    ],
                    dtype=np.float64,
                )

                if (
                    np.std(
                        abs_pe_values,
                        ddof=0,
                    )
                    < EPSILON
                ):
                    raise ValueError(
                        "abs_vPE_dual has zero or near-zero variance "
                        "within the included trials."
                    )

                seed_z = zscore_vector(
                    seed_array
                )

                abs_pe_z = zscore_vector(
                    abs_pe_values
                )

                # Whole-GM nuisance activity is extracted from the
                # outcome-locked LSS beta maps.
                gm_z = zscore_vector(
                    gm_beta_array
                )

                if arguments.residualize_seed_by_gm:
                    seed_residual_design = np.column_stack(
                        (
                            gm_z,
                            np.ones_like(
                                gm_z
                            ),
                        )
                    )

                    seed_coefficients, _, seed_rank, _ = (
                        np.linalg.lstsq(
                            seed_residual_design,
                            seed_z,
                            rcond=None,
                        )
                    )

                    if seed_rank < 2:
                        raise RuntimeError(
                            "The seed residualization design is "
                            "rank deficient."
                        )

                    seed_z = (
                        seed_z
                        - seed_residual_design
                        @ seed_coefficients
                    )

                    seed_z = zscore_vector(
                        seed_z
                    )

                interaction_z = zscore_vector(
                    seed_z
                    * abs_pe_z
                )

                design_matrix = np.column_stack(
                    (
                        seed_z,
                        abs_pe_z,
                        interaction_z,
                        gm_z,
                        np.ones_like(
                            seed_z
                        ),
                    )
                )

                regressor_names = [
                    "seed",
                    "absPE",
                    "seedXabsPE",
                    "GM",
                    "intercept",
                ]

                number_of_trials = int(
                    design_matrix.shape[0]
                )

                number_of_parameters = int(
                    design_matrix.shape[1]
                )

                design_rank = int(
                    np.linalg.matrix_rank(
                        design_matrix
                    )
                )

                degrees_of_freedom = int(
                    number_of_trials
                    - design_rank
                )

                condition_number = float(
                    np.linalg.cond(
                        design_matrix
                    )
                )

                base_qc[
                    "rank"
                ] = design_rank

                base_qc[
                    "dof"
                ] = degrees_of_freedom

                base_qc[
                    "condition_number_X"
                ] = condition_number

                if design_rank < number_of_parameters:
                    raise RuntimeError(
                        "The design matrix is rank deficient: "
                        f"rank={design_rank}, "
                        f"parameters={number_of_parameters}."
                    )

                if (
                    degrees_of_freedom
                    < arguments.minimum_dof
                ):
                    raise RuntimeError(
                        "Residual degrees of freedom are too low: "
                        f"{degrees_of_freedom} < "
                        f"{arguments.minimum_dof}."
                    )

                if not np.isfinite(
                    condition_number
                ):
                    raise RuntimeError(
                        "The design-matrix condition number is not finite."
                    )

                if (
                    condition_number
                    > arguments.condition_number_warning
                ):
                    print(
                        f"[Warning] {subject_name}/{roi_name}: "
                        "high design-matrix condition number "
                        f"({condition_number:.3g})."
                    )

                dependent_matrix = pattern_stack.reshape(
                    number_of_trials,
                    -1,
                )

                gm_mask_flat = gm_mask.reshape(
                    -1
                )

                finite_voxels = np.all(
                    np.isfinite(
                        dependent_matrix
                    ),
                    axis=0,
                )

                valid_voxels = (
                    gm_mask_flat
                    & finite_voxels
                )

                dependent_valid = dependent_matrix[
                    :,
                    valid_voxels,
                ]

                if dependent_valid.size == 0:
                    raise RuntimeError(
                        "No finite dependent-variable voxels remained "
                        "within the whole-GM mask."
                    )

                design_pseudoinverse = np.linalg.pinv(
                    design_matrix
                )

                beta_matrix = (
                    design_pseudoinverse
                    @ dependent_valid
                )

                residual_matrix = (
                    dependent_valid
                    - design_matrix
                    @ beta_matrix
                )

                sigma_squared = (
                    np.sum(
                        residual_matrix
                        * residual_matrix,
                        axis=0,
                    )
                    / degrees_of_freedom
                )

                sigma_squared = np.maximum(
                    sigma_squared,
                    0.0,
                )

                xtx_inverse = np.linalg.pinv(
                    design_matrix.T
                    @ design_matrix
                )

                subject_output_directory = (
                    roi_output_directory
                    / subject_name
                )

                subject_output_directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                if not arguments.skip_design_csv:
                    design_table = pd.DataFrame(
                        {
                            "subject": subject_name,
                            "roi": roi_name,
                            "global_trial": [
                                row["global_trial"]
                                for row in used_pairs
                            ],
                            "block": [
                                row["block"]
                                for row in used_pairs
                            ],
                            "trial_in_block": [
                                row["trial_in_block"]
                                for row in used_pairs
                            ],
                            "reward_raw": [
                                row["reward_raw"]
                                for row in used_pairs
                            ],
                            "reward_model": [
                                row["reward_model"]
                                for row in used_pairs
                            ],
                            "no_reward": [
                                row["no_reward"]
                                for row in used_pairs
                            ],
                            "vPE_dual": vpe_values,
                            "abs_vPE_dual": abs_pe_values,
                            "absPE_z": abs_pe_z,
                            "seed_raw": seed_array,
                            "seed_z": seed_z,
                            "seedXabsPE_z": interaction_z,
                            "gm_beta_raw": gm_beta_array,
                            "gm_beta_z": gm_z,
                            "pattern_file": [
                                relative_path_text(
                                    row["pattern_path"],
                                    pattern_root,
                                )
                                for row in used_pairs
                            ],
                            "beta_file": [
                                relative_path_text(
                                    row["beta_path"],
                                    lss_root,
                                )
                                for row in used_pairs
                            ],
                        }
                    )

                    design_path = (
                        subject_output_directory
                        / (
                            f"{subject_name}_{roi_name}_"
                            "RN_noRewardMatched_Dual_design.csv"
                        )
                    )

                    design_table.to_csv(
                        design_path,
                        index=False,
                        encoding="utf-8",
                    )

                beta_outputs = {
                    "seed": beta_matrix[
                        0,
                        :,
                    ],
                    "absPE": beta_matrix[
                        1,
                        :,
                    ],
                    "seedXabsPE": beta_matrix[
                        2,
                        :,
                    ],
                    "GM": beta_matrix[
                        3,
                        :,
                    ],
                }

                for (
                    regressor_name,
                    regressor_values,
                ) in beta_outputs.items():
                    output_path = (
                        subject_output_directory
                        / (
                            f"{subject_name}_{roi_name}_RN_"
                            "noRewardMatched_Dual_"
                            f"{regressor_name}"
                            f"{arguments.output_extension}"
                        )
                    )

                    save_flat_map(
                        values_valid=regressor_values,
                        valid_mask_flat=valid_voxels,
                        volume_shape=volume_shape,
                        reference_image=(
                            pattern_reference_image
                        ),
                        output_path=output_path,
                    )

                    map_manifest_rows.append(
                        {
                            "subject": subject_name,
                            "roi": roi_name,
                            "regressor": regressor_name,
                            "map_path": relative_path_text(
                                output_path,
                                output_root,
                            ),
                        }
                    )

                    print(
                        f"[Saved] {subject_name}/{roi_name}/"
                        f"{regressor_name}"
                    )

                if arguments.save_tmaps:
                    for (
                        regressor_index,
                        regressor_name,
                    ) in enumerate(
                        regressor_names[
                            :4
                        ]
                    ):
                        beta_variance = (
                            sigma_squared
                            * xtx_inverse[
                                regressor_index,
                                regressor_index,
                            ]
                        )

                        standard_error = np.sqrt(
                            np.maximum(
                                beta_variance,
                                EPSILON,
                            )
                        )

                        t_values = (
                            beta_matrix[
                                regressor_index,
                                :,
                            ]
                            / standard_error
                        )

                        tmap_path = (
                            subject_output_directory
                            / (
                                f"{subject_name}_{roi_name}_RN_"
                                "noRewardMatched_Dual_"
                                f"{regressor_name}_tmap"
                                f"{arguments.output_extension}"
                            )
                        )

                        save_flat_map(
                            values_valid=t_values,
                            valid_mask_flat=valid_voxels,
                            volume_shape=volume_shape,
                            reference_image=(
                                pattern_reference_image
                            ),
                            output_path=tmap_path,
                        )

                base_qc[
                    "status"
                ] = "success"

                base_qc[
                    "mean_absPE"
                ] = float(
                    np.mean(
                        abs_pe_values
                    )
                )

                base_qc[
                    "std_absPE"
                ] = float(
                    np.std(
                        abs_pe_values,
                        ddof=0,
                    )
                )

                base_qc[
                    "min_absPE"
                ] = float(
                    np.min(
                        abs_pe_values
                    )
                )

                base_qc[
                    "max_absPE"
                ] = float(
                    np.max(
                        abs_pe_values
                    )
                )

                base_qc[
                    "mean_seed"
                ] = float(
                    np.mean(
                        seed_array
                    )
                )

                base_qc[
                    "std_seed"
                ] = float(
                    np.std(
                        seed_array,
                        ddof=0,
                    )
                )

                base_qc[
                    "mean_GM_beta"
                ] = float(
                    np.mean(
                        gm_beta_array
                    )
                )

                base_qc[
                    "std_GM_beta"
                ] = float(
                    np.std(
                        gm_beta_array,
                        ddof=0,
                    )
                )

                base_qc[
                    "corr_seed_absPE"
                ] = safe_correlation(
                    seed_z,
                    abs_pe_z,
                )

                base_qc[
                    "corr_absPE_GM"
                ] = safe_correlation(
                    abs_pe_z,
                    gm_z,
                )

                base_qc[
                    "corr_seed_GM"
                ] = safe_correlation(
                    seed_z,
                    gm_z,
                )

                print(
                    f"[Completed] {subject_name}/{roi_name}: "
                    f"trials={number_of_trials}, "
                    f"dof={degrees_of_freedom}, "
                    f"condition={condition_number:.3g}"
                )

            except Exception as error:
                base_qc[
                    "status"
                ] = "failed"

                base_qc[
                    "error_message"
                ] = sanitize_error_message(
                    error,
                    replacement_roots,
                )

                print(
                    f"[Failed] {subject_name}/{roi_name}: {error}"
                )

            qc_rows.append(
                base_qc
            )

    qc_table = pd.DataFrame(
        qc_rows
    )

    qc_path = (
        output_root
        / "QC_RN_noRewardMatched_Dual.csv"
    )

    qc_table.to_csv(
        qc_path,
        index=False,
        encoding="utf-8",
    )

    successful_qc = qc_table.loc[
        qc_table["status"] == "success"
    ].copy()

    successful_qc_path = (
        output_root
        / "QC_RN_noRewardMatched_Dual_success_only.csv"
    )

    successful_qc.to_csv(
        successful_qc_path,
        index=False,
        encoding="utf-8",
    )

    manifest_columns = [
        "subject",
        "roi",
        "regressor",
        "map_path",
    ]

    map_manifest = pd.DataFrame(
        map_manifest_rows,
        columns=manifest_columns,
    )

    manifest_path = (
        output_root
        / "second_level_beta_map_manifest.csv"
    )

    map_manifest.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8",
    )

    if not map_manifest.empty:
        list_root = (
            output_root
            / "second_level_input_lists"
        )

        list_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        for (
            roi_name,
            regressor_name,
        ), group_table in map_manifest.groupby(
            [
                "roi",
                "regressor",
            ]
        ):
            group_table = group_table.copy()

            group_table[
                "_subject_number"
            ] = group_table[
                "subject"
            ].map(
                subject_numeric_key
            )

            group_table = group_table.sort_values(
                by=[
                    "_subject_number",
                ]
            )

            list_path = (
                list_root
                / (
                    f"{roi_name}_"
                    f"{regressor_name}_maps.txt"
                )
            )

            list_path.write_text(
                "\n".join(
                    group_table[
                        "map_path"
                    ].tolist()
                ),
                encoding="utf-8",
            )

    number_successful = int(
        (
            qc_table["status"]
            == "success"
        ).sum()
    )

    number_failed = int(
        (
            qc_table["status"]
            == "failed"
        ).sum()
    )

    print("\n" + "=" * 80)
    print("Analysis completed")
    print("=" * 80)

    print(
        f"Successful models:       {number_successful}"
    )
    print(
        f"Failed models:           {number_failed}"
    )
    print(
        f"QC report:               {qc_path}"
    )
    print(
        f"Successful-model report: {successful_qc_path}"
    )
    print(
        f"Second-level manifest:   {manifest_path}"
    )
    print(
        f"Output root:             {output_root}"
    )


if __name__ == "__main__":
    main()