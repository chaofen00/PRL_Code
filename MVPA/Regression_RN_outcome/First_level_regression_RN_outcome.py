#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
First-level ROI-to-pattern-change regression for positive and negative vPE
trials during the RN stage.

For each participant, ROI, and vPE-sign group:

1. Identify RN trials using cue_filtered_trials.csv.
2. Align RN trials with trial-wise vPE estimates.
3. Divide trials into:
       RN_pos: vPE > 0
       RN_neg: vPE < 0
   Trials with vPE == 0 or non-finite vPE values are excluded.
4. For each included trial, load the corresponding outcome-locked LSS beta
   map and pattern-change map.
5. Extract from the LSS beta maps:
       - the mean ROI beta series;
       - the mean whole-GM beta series.
6. Within each participant and vPE-sign group:
       - z-score the ROI beta series;
       - z-score the whole-GM beta series;
       - regress the ROI beta series on the whole-GM beta series;
       - z-score the residualized ROI series.
   No temporal detrending is performed.
7. Use the residualized ROI series as the predictor in a voxel-wise OLS
   regression of the corresponding pattern-change maps.
8. Save the voxel-wise slope map for second-level analysis.

Important
---------
The whole-GM nuisance series is calculated from the outcome-locked LSS
beta maps, not from the pattern-change maps.

Pattern-change maps are used as:
    1. trial-level voxel-wise outcome data;
    2. indicators of which labeled trial maps are available.

Expected directory structure
----------------------------
project/
├── run_rn_vpe_sign_regression.py
├── data/
│   ├── rois/
│   │   ├── roi_1.nii
│   │   └── roi_2.nii
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
│   ├── qlearning/
│   │   ├── Qlearning_trialwise_02.csv
│   │   └── ...
│   └── masks/
│       └── group_gm_mask.nii
└── outputs/
    └── rn_vpe_sign_regression/

Output structure
----------------
<output_root>/
├── ROI_1/
│   ├── Sub02/
│   │   ├── Sub02_ROI_1_regBeta_GMresid_RN_pos.nii
│   │   └── Sub02_ROI_1_regBeta_GMresid_RN_neg.nii
│   └── ...
├── ROI_2/
│   └── ...
└── first_level_vpe_sign_summary.tsv

Dependencies
------------
numpy
nibabel
pandas
nilearn  (optional; used only for new_img_like)

Example
-------
python run_rn_vpe_sign_regression.py \
    --roi-dir /path/to/rois \
    --lss-root /path/to/lss_outcome \
    --pattern-root /path/to/pattern_change_labeled \
    --onsets-root /path/to/onsets \
    --q-root /path/to/qlearning \
    --gm-mask /path/to/group_gm_mask.nii \
    --output-root /path/to/output
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import pandas as pd

try:
    from nilearn.image import new_img_like

    USE_NILEARN = True
except ImportError:
    USE_NILEARN = False


TARGET_LABEL = "RN"

BETA_PREFIX = "beta_outcome_"
PATTERN_PREFIX = "pattern_change_trial_"

SIGN_NAMES = (
    "pos",
    "neg",
)

NIFTI_SUFFIXES = (
    ".nii",
    ".nii.gz",
)

EPSILON = 1e-6


def parse_arguments() -> argparse.Namespace:
    """Parse portable path and analysis settings."""
    script_directory = Path(__file__).resolve().parent
    default_data_root = script_directory / "data"

    parser = argparse.ArgumentParser(
        description=(
            "Run RN-stage voxel-wise pattern-change regressions separately "
            "for positive and negative vPE trials."
        )
    )

    parser.add_argument(
        "--roi-dir",
        type=Path,
        default=default_data_root / "rois",
        help=(
            "Directory containing ROI mask images. "
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
        default=default_data_root / "qlearning",
        help=(
            "Directory containing Qlearning_trialwise_*.csv files. "
            "Default: <script_dir>/data/qlearning"
        ),
    )

    parser.add_argument(
        "--gm-mask",
        type=Path,
        default=default_data_root / "masks" / "group_gm_mask.nii",
        help=(
            "Whole-GM mask used to extract trial-wise GM beta values. "
            "Default: <script_dir>/data/masks/group_gm_mask.nii"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=script_directory / "outputs" / "rn_vpe_sign_regression",
        help=(
            "Output root for participant-level slope maps. "
            "Default: <script_dir>/outputs/rn_vpe_sign_regression"
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
        default=8,
        help=(
            "Minimum number of aligned trials required within each "
            "vPE-sign group. Default: 8"
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
        "--output-extension",
        choices=(".nii", ".nii.gz"),
        default=".nii",
        help="Output image extension. Default: .nii",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip analyses whose output slope map already exists.",
    )

    return parser.parse_args()


def natural_sort_key(value: str) -> list[object]:
    """Sort names naturally, for example Sub2 before Sub10."""
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
        and path.name.lower().endswith(NIFTI_SUFFIXES)
    )


def clean_tsv_field(value: object) -> str:
    """Remove tabs and line breaks before writing a TSV field."""
    return (
        str(value)
        .replace("\t", " ")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def load_3d_image(
    image_path: Path,
) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
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
            f"{image_path}"
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
    """Raise an error if two images do not share the same voxel grid."""
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


def first_existing_nifti(
    base_path: Path,
) -> Path | None:
    """Return an existing .nii or .nii.gz version of a path."""
    if base_path.is_file():
        return base_path

    path_text = str(base_path)

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


def trial_from_filename(
    filename: str,
    prefix: str,
) -> int | None:
    """Extract a trial number from a prefixed NIfTI filename."""
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
    """Locate a participant's outcome-locked LSS directory."""
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
    """Locate a participant's trial-wise Q-learning CSV file."""
    subject_id = normalize_subject_id(
        subject_name
    )

    participant_number = int(
        subject_id
    )

    candidate_names = [
        f"Qlearning_trialwise_{subject_id}.csv",
        f"Qlearning_trialwise_{participant_number}.csv",
        f"Qlearning_trialwise_{participant_number:02d}.csv",
        f"Qlearning_trialwise_{participant_number:03d}.csv",
    ]

    checked_names: set[str] = set()

    for candidate_name in candidate_names:
        if candidate_name in checked_names:
            continue

        checked_names.add(
            candidate_name
        )

        candidate_path = (
            q_root
            / candidate_name
        )

        if candidate_path.is_file():
            return candidate_path

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
                f"[Missing] Pattern-change directory was not found "
                f"for participant {requested_id}."
            )
            continue

        selected_directories.append(
            subject_directory
        )

    unique_directories = {
        path.resolve(): path
        for path in selected_directories
    }

    return sorted(
        unique_directories.values(),
        key=lambda path: natural_sort_key(
            path.name
        ),
    )


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
            f"Could not decode CSV file: {csv_path}. "
            f"Last error: {last_error}"
        )

    return dataframe


def read_rn_trial_vpe_mapping(
    label_csv_path: Path,
    qlearning_csv_path: Path,
) -> dict[int, float]:
    """
    Align RN trials with trial-wise vPE values.

    The label CSV is matched to the Q-learning table using:

        trial <-> trial_in_block
    """
    label_table = read_csv_with_fallback(
        label_csv_path
    )

    label_columns = {
        str(column).strip().lower(): column
        for column in label_table.columns
    }

    if (
        "trial" not in label_columns
        or "label" not in label_columns
    ):
        raise ValueError(
            f"Label CSV lacks trial or label columns: {label_csv_path}"
        )

    label_table = label_table.rename(
        columns={
            label_columns["trial"]: "trial",
            label_columns["label"]: "label",
        }
    )

    label_table = label_table[
        [
            "trial",
            "label",
        ]
    ].copy()

    label_table["trial"] = pd.to_numeric(
        label_table["trial"],
        errors="coerce",
    )

    label_table["label"] = (
        label_table["label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    label_table = label_table.dropna(
        subset=[
            "trial",
        ]
    ).copy()

    label_table["trial"] = (
        label_table["trial"]
        .astype(int)
    )

    label_table = label_table.loc[
        label_table["trial"] >= 1
    ].copy()

    label_table = label_table.loc[
        label_table["label"] == TARGET_LABEL
    ].copy()

    if label_table.empty:
        return {}

    label_conflicts = (
        label_table
        .groupby("trial")["label"]
        .nunique()
    )

    conflicting_label_trials = (
        label_conflicts[
            label_conflicts > 1
        ]
        .index
        .tolist()
    )

    if conflicting_label_trials:
        raise ValueError(
            "The label CSV assigns conflicting labels to trials: "
            f"{conflicting_label_trials[:20]}"
        )

    label_table = label_table.drop_duplicates(
        subset=[
            "trial",
        ],
        keep="first",
    )

    qlearning_table = read_csv_with_fallback(
        qlearning_csv_path
    )

    qlearning_columns = {
        str(column).strip().lower(): column
        for column in qlearning_table.columns
    }

    if (
        "trial_in_block" not in qlearning_columns
        or "vpe" not in qlearning_columns
    ):
        raise ValueError(
            "The Q-learning CSV lacks trial_in_block or vPE columns: "
            f"{qlearning_csv_path}"
        )

    qlearning_table = qlearning_table.rename(
        columns={
            qlearning_columns["trial_in_block"]: "trial_in_block",
            qlearning_columns["vpe"]: "vpe",
        }
    )

    qlearning_table = qlearning_table[
        [
            "trial_in_block",
            "vpe",
        ]
    ].copy()

    qlearning_table["trial_in_block"] = pd.to_numeric(
        qlearning_table["trial_in_block"],
        errors="coerce",
    )

    qlearning_table["vpe"] = pd.to_numeric(
        qlearning_table["vpe"],
        errors="coerce",
    )

    qlearning_table = qlearning_table.dropna(
        subset=[
            "trial_in_block",
        ]
    ).copy()

    qlearning_table["trial_in_block"] = (
        qlearning_table["trial_in_block"]
        .astype(int)
    )

    # Prevent silently assigning different vPE values to the same trial.
    qlearning_conflicts = (
        qlearning_table
        .dropna(subset=["vpe"])
        .groupby("trial_in_block")["vpe"]
        .nunique()
    )

    conflicting_q_trials = (
        qlearning_conflicts[
            qlearning_conflicts > 1
        ]
        .index
        .tolist()
    )

    if conflicting_q_trials:
        raise ValueError(
            "The Q-learning CSV contains multiple vPE values for the "
            "same trial_in_block: "
            f"{conflicting_q_trials[:20]}"
        )

    qlearning_table = qlearning_table.drop_duplicates(
        subset=[
            "trial_in_block",
        ],
        keep="first",
    )

    merged_table = label_table.merge(
        qlearning_table,
        left_on="trial",
        right_on="trial_in_block",
        how="left",
        validate="one_to_one",
    )

    merged_table = merged_table.dropna(
        subset=[
            "vpe",
        ]
    ).copy()

    return {
        int(row.trial): float(row.vpe)
        for row in merged_table.itertuples(
            index=False
        )
        if np.isfinite(row.vpe)
    }


def list_pattern_files(
    label_directory: Path,
) -> list[Path]:
    """List pattern-change maps sorted by trial number."""
    files_by_trial: dict[int, Path] = {}

    if not label_directory.is_dir():
        return []

    for image_path in label_directory.iterdir():
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
            files_by_trial[trial_number] = image_path
            continue

        if (
            image_path.name.lower().endswith(".nii.gz")
            and not existing_path.name.lower().endswith(".nii.gz")
        ):
            files_by_trial[trial_number] = image_path

    return [
        files_by_trial[trial_number]
        for trial_number in sorted(
            files_by_trial
        )
    ]


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
            if is_nifti_file(path)
        ],
        key=lambda path: natural_sort_key(
            path.name
        ),
    )

    if not roi_files:
        raise FileNotFoundError(
            f"No ROI NIfTI files were found in: {roi_directory}"
        )

    roi_masks = {}

    for roi_path in roi_files:
        roi_image, roi_data = load_3d_image(
            roi_path
        )

        roi_mask = (
            np.isfinite(roi_data)
            & (roi_data > 0)
        )

        if not np.any(
            roi_mask
        ):
            print(
                f"[Warning] Empty ROI mask skipped: {roi_path}"
            )
            continue

        roi_name = strip_nifti_suffix(
            roi_path
        )

        if roi_name in roi_masks:
            raise ValueError(
                f"Duplicated ROI name after suffix removal: {roi_name}"
            )

        roi_masks[roi_name] = (
            roi_image,
            roi_mask,
        )

    if not roi_masks:
        raise RuntimeError(
            "All ROI masks were empty or invalid."
        )

    return roi_masks


def build_vpe_sign_groups(
    subject_directory: Path,
    lss_directory: Path,
    trial_to_vpe: dict[int, float],
) -> tuple[
    dict[
        str,
        list[
            tuple[
                int,
                Path,
                Path,
            ]
        ],
    ],
    int,
]:
    """
    Match RN pattern-change maps with LSS beta maps and divide them by vPE.

    Returns
    -------
    sign_groups:
        Dictionary containing positive- and negative-vPE trial tuples.

    number_of_pattern_files:
        Total number of RN pattern-change maps found.
    """
    rn_directory = (
        subject_directory
        / TARGET_LABEL
    )

    pattern_files = list_pattern_files(
        rn_directory
    )

    sign_groups: dict[
        str,
        list[
            tuple[
                int,
                Path,
                Path,
            ]
        ],
    ] = {
        "pos": [],
        "neg": [],
    }

    for pattern_path in pattern_files:
        trial_number = trial_from_filename(
            pattern_path.name,
            PATTERN_PREFIX,
        )

        if trial_number is None:
            continue

        if trial_number not in trial_to_vpe:
            continue

        vpe_value = trial_to_vpe[
            trial_number
        ]

        if (
            not np.isfinite(vpe_value)
            or vpe_value == 0
        ):
            continue

        beta_path = first_existing_nifti(
            lss_directory
            / f"{BETA_PREFIX}{trial_number:03d}.nii"
        )

        if beta_path is None:
            print(
                f"[Missing] {subject_directory.name}/{TARGET_LABEL}, "
                f"trial={trial_number:03d}: outcome LSS beta not found."
            )
            continue

        group_name = (
            "pos"
            if vpe_value > 0
            else "neg"
        )

        sign_groups[group_name].append(
            (
                trial_number,
                pattern_path,
                beta_path,
            )
        )

    for group_name in SIGN_NAMES:
        sign_groups[group_name].sort(
            key=lambda item: item[0]
        )

    return (
        sign_groups,
        len(pattern_files),
    )


def load_aligned_trial_images(
    aligned_trials: Sequence[
        tuple[
            int,
            Path,
            Path,
        ]
    ],
    gm_image: nib.spatialimages.SpatialImage,
    affine_tolerance: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    nib.spatialimages.SpatialImage,
    nib.spatialimages.SpatialImage,
]:
    """
    Load aligned LSS beta maps and pattern-change maps.

    Returns
    -------
    beta_stack:
        Outcome-locked LSS beta maps with shape T × X × Y × Z.

    pattern_stack:
        Pattern-change maps with shape T × X × Y × Z.

    beta_reference:
        Reference beta image used for mask validation.

    pattern_reference:
        Reference pattern-change image used to save the slope map.
    """
    beta_arrays: list[np.ndarray] = []
    pattern_arrays: list[np.ndarray] = []

    beta_reference = None
    beta_reference_path = None

    pattern_reference = None
    pattern_reference_path = None

    for (
        _,
        pattern_path,
        beta_path,
    ) in aligned_trials:
        beta_image, beta_data = load_3d_image(
            beta_path
        )

        pattern_image, pattern_data = load_3d_image(
            pattern_path
        )

        if beta_reference is None:
            beta_reference = beta_image
            beta_reference_path = beta_path

            require_same_grid(
                beta_reference,
                gm_image,
                str(beta_reference_path),
                "whole-GM mask",
                affine_tolerance,
            )

        else:
            require_same_grid(
                beta_reference,
                beta_image,
                str(beta_reference_path),
                str(beta_path),
                affine_tolerance,
            )

        if pattern_reference is None:
            pattern_reference = pattern_image
            pattern_reference_path = pattern_path

        else:
            require_same_grid(
                pattern_reference,
                pattern_image,
                str(pattern_reference_path),
                str(pattern_path),
                affine_tolerance,
            )

        require_same_grid(
            beta_image,
            pattern_image,
            str(beta_path),
            str(pattern_path),
            affine_tolerance,
        )

        beta_arrays.append(
            beta_data
        )

        pattern_arrays.append(
            pattern_data
        )

    if (
        beta_reference is None
        or pattern_reference is None
    ):
        raise ValueError(
            "No aligned trial images were loaded."
        )

    beta_stack = np.stack(
        beta_arrays,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    pattern_stack = np.stack(
        pattern_arrays,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    return (
        beta_stack,
        pattern_stack,
        beta_reference,
        pattern_reference,
    )


def residualize_roi_series(
    roi_z: np.ndarray,
    gm_z: np.ndarray,
) -> np.ndarray:
    """
    Remove whole-GM beta variation from the ROI beta series.

    Model
    -----
    ROI_z = intercept + coefficient * GM_z + residual
    """
    design_matrix = np.column_stack(
        (
            gm_z,
            np.ones_like(gm_z),
        )
    )

    coefficients, _, design_rank, _ = np.linalg.lstsq(
        design_matrix,
        roi_z,
        rcond=None,
    )

    if design_rank < 2:
        raise ValueError(
            "The whole-GM residualization model is rank deficient."
        )

    residuals = (
        roi_z
        - design_matrix @ coefficients
    )

    return zscore_vector(
        residuals
    )


def fit_voxelwise_slopes(
    predictor: np.ndarray,
    pattern_stack: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Fit voxel-wise OLS slopes.

    Model
    -----
    pattern_change = intercept + slope * residualized_ROI_predictor

    Only voxels with finite pattern-change values across all included
    trials are entered into the regression.
    """
    number_of_trials = predictor.size

    voxel_matrix = pattern_stack.reshape(
        number_of_trials,
        -1,
    )

    valid_voxels = np.all(
        np.isfinite(voxel_matrix),
        axis=0,
    )

    number_of_valid_voxels = int(
        valid_voxels.sum()
    )

    if number_of_valid_voxels == 0:
        raise ValueError(
            "No voxel has finite pattern-change values across all trials."
        )

    valid_voxel_data = voxel_matrix[
        :,
        valid_voxels,
    ]

    design_matrix = np.column_stack(
        (
            predictor,
            np.ones_like(predictor),
        )
    )

    coefficients, _, design_rank, _ = np.linalg.lstsq(
        design_matrix,
        valid_voxel_data,
        rcond=None,
    )

    if design_rank < 2:
        raise ValueError(
            "The voxel-wise regression design is rank deficient."
        )

    slope_values = coefficients[
        0,
        :
    ]

    slope_flat = np.full(
        voxel_matrix.shape[1],
        np.nan,
        dtype=np.float32,
    )

    slope_flat[
        valid_voxels
    ] = slope_values.astype(
        np.float32
    )

    slope_volume = slope_flat.reshape(
        pattern_stack.shape[1:]
    )

    return (
        slope_volume,
        number_of_valid_voxels,
    )


def write_summary(
    output_root: Path,
    summary_rows: list[dict[str, object]],
) -> Path:
    """Write a TSV summary of all requested first-level analyses."""
    summary_path = (
        output_root
        / "first_level_vpe_sign_summary.tsv"
    )

    columns = [
        "roi",
        "participant_id",
        "stage",
        "vpe_group",
        "n_pattern_files",
        "n_aligned_trials",
        "n_valid_voxels",
        "output_file",
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
                    row.get(column, "")
                )
                for column in columns
            ]

            file.write(
                "\t".join(values)
                + "\n"
            )

    return summary_path


def main() -> None:
    """Run RN positive- and negative-vPE first-level regressions."""
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

    qlearning_root = (
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
        "Q-learning root": qlearning_root,
    }

    for description, directory_path in required_directories.items():
        if not directory_path.is_dir():
            raise NotADirectoryError(
                f"{description} does not exist: {directory_path}"
            )

    if not gm_mask_path.is_file():
        raise FileNotFoundError(
            f"Whole-GM mask does not exist: {gm_mask_path}"
        )

    if arguments.minimum_trials < 3:
        raise ValueError(
            "--minimum-trials must be at least 3."
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
        np.isfinite(gm_data)
        & (gm_data > 0)
    )

    if not np.any(
        gm_mask
    ):
        raise ValueError(
            f"The whole-GM mask is empty: {gm_mask_path}"
        )

    subject_directories = get_subject_directories(
        pattern_root,
        arguments.subject_ids,
    )

    if not subject_directories:
        raise RuntimeError(
            "No participant pattern-change directories were found."
        )

    print(f"ROI directory:         {roi_directory}")
    print(f"LSS root:              {lss_root}")
    print(f"Pattern-change root:   {pattern_root}")
    print(f"Onsets root:           {onsets_root}")
    print(f"Q-learning root:       {qlearning_root}")
    print(f"Whole-GM mask:         {gm_mask_path}")
    print(f"Output root:           {output_root}")
    print(f"ROIs:                  {len(roi_masks)}")
    print(f"Participants:          {len(subject_directories)}")
    print(f"Target stage:          {TARGET_LABEL}")
    print(f"Minimum trials/group:  {arguments.minimum_trials}")
    print(f"Skip existing:         {arguments.skip_existing}")

    summary_rows: list[dict[str, object]] = []

    # Process participant and vPE group first so that each image stack can
    # be reused across all ROI analyses.
    for subject_directory in subject_directories:
        subject_name = subject_directory.name
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

        qlearning_path = find_qlearning_file(
            qlearning_root,
            subject_name,
        )

        if (
            lss_directory is None
            or onset_subject_directory is None
            or qlearning_path is None
        ):
            reason = (
                "LSS directory, onset directory, or Q-learning CSV "
                "was not found"
            )

            print(
                f"[Skipped] {subject_name}: {reason}."
            )

            for roi_name in roi_masks:
                for sign_name in SIGN_NAMES:
                    summary_rows.append(
                        {
                            "roi": roi_name,
                            "participant_id": subject_name,
                            "stage": TARGET_LABEL,
                            "vpe_group": sign_name,
                            "status": "skipped",
                            "reason": reason,
                        }
                    )

            continue

        label_csv_path = (
            onset_subject_directory
            / "csv_output"
            / "cue_filtered_trials.csv"
        )

        if not label_csv_path.is_file():
            reason = "cue_filtered_trials.csv was not found"

            print(
                f"[Skipped] {subject_name}: {reason}."
            )

            for roi_name in roi_masks:
                for sign_name in SIGN_NAMES:
                    summary_rows.append(
                        {
                            "roi": roi_name,
                            "participant_id": subject_name,
                            "stage": TARGET_LABEL,
                            "vpe_group": sign_name,
                            "status": "skipped",
                            "reason": reason,
                        }
                    )

            continue

        try:
            trial_to_vpe = read_rn_trial_vpe_mapping(
                label_csv_path,
                qlearning_path,
            )

        except Exception as error:
            print(
                f"[Skipped] {subject_name}: {error}"
            )

            for roi_name in roi_masks:
                for sign_name in SIGN_NAMES:
                    summary_rows.append(
                        {
                            "roi": roi_name,
                            "participant_id": subject_name,
                            "stage": TARGET_LABEL,
                            "vpe_group": sign_name,
                            "status": "skipped",
                            "reason": str(error),
                        }
                    )

            continue

        if not trial_to_vpe:
            reason = "No RN trial could be aligned with a finite vPE value"

            print(
                f"[Skipped] {subject_name}: {reason}."
            )

            for roi_name in roi_masks:
                for sign_name in SIGN_NAMES:
                    summary_rows.append(
                        {
                            "roi": roi_name,
                            "participant_id": subject_name,
                            "stage": TARGET_LABEL,
                            "vpe_group": sign_name,
                            "status": "skipped",
                            "reason": reason,
                        }
                    )

            continue

        (
            sign_groups,
            number_of_pattern_files,
        ) = build_vpe_sign_groups(
            subject_directory=subject_directory,
            lss_directory=lss_directory,
            trial_to_vpe=trial_to_vpe,
        )

        for sign_name in SIGN_NAMES:
            aligned_trials = sign_groups[
                sign_name
            ]

            number_of_aligned_trials = len(
                aligned_trials
            )

            if number_of_aligned_trials < arguments.minimum_trials:
                reason = (
                    f"too few aligned trials "
                    f"({number_of_aligned_trials} < "
                    f"{arguments.minimum_trials})"
                )

                print(
                    f"[Skipped] {subject_name}/"
                    f"{TARGET_LABEL}_{sign_name}: {reason}."
                )

                for roi_name in roi_masks:
                    summary_rows.append(
                        {
                            "roi": roi_name,
                            "participant_id": subject_name,
                            "stage": TARGET_LABEL,
                            "vpe_group": sign_name,
                            "n_pattern_files": number_of_pattern_files,
                            "n_aligned_trials": number_of_aligned_trials,
                            "status": "skipped",
                            "reason": reason,
                        }
                    )

                continue

            print(
                f"\n[Loading] {subject_name}/"
                f"{TARGET_LABEL}_{sign_name}: "
                f"{number_of_aligned_trials} aligned trials."
            )

            try:
                (
                    beta_stack,
                    pattern_stack,
                    beta_reference,
                    pattern_reference,
                ) = load_aligned_trial_images(
                    aligned_trials=aligned_trials,
                    gm_image=gm_image,
                    affine_tolerance=arguments.affine_tolerance,
                )

                if gm_mask.shape != beta_stack.shape[1:]:
                    raise ValueError(
                        "The whole-GM mask shape does not match the "
                        f"LSS beta maps: mask={gm_mask.shape}, "
                        f"beta={beta_stack.shape[1:]}"
                    )

                # Extract the nuisance series from the outcome-locked LSS
                # beta maps, not from the pattern-change maps.
                gm_trial_series = np.nanmean(
                    beta_stack[:, gm_mask],
                    axis=1,
                )

                gm_z = zscore_vector(
                    gm_trial_series
                )

            except Exception as error:
                print(
                    f"[Failed] {subject_name}/"
                    f"{TARGET_LABEL}_{sign_name}: {error}"
                )

                for roi_name in roi_masks:
                    summary_rows.append(
                        {
                            "roi": roi_name,
                            "participant_id": subject_name,
                            "stage": TARGET_LABEL,
                            "vpe_group": sign_name,
                            "n_pattern_files": number_of_pattern_files,
                            "n_aligned_trials": number_of_aligned_trials,
                            "status": "failed",
                            "reason": str(error),
                        }
                    )

                continue

            for roi_name, (
                roi_image,
                roi_mask,
            ) in roi_masks.items():
                output_subject_directory = (
                    output_root
                    / roi_name
                    / subject_name
                )

                output_path = (
                    output_subject_directory
                    / (
                        f"{subject_name}_{roi_name}_"
                        f"regBeta_GMresid_"
                        f"{TARGET_LABEL}_{sign_name}"
                        f"{arguments.output_extension}"
                    )
                )

                if (
                    arguments.skip_existing
                    and output_path.is_file()
                ):
                    print(
                        f"[Existing] {subject_name}/"
                        f"{roi_name}/{TARGET_LABEL}_{sign_name}"
                    )

                    summary_rows.append(
                        {
                            "roi": roi_name,
                            "participant_id": subject_name,
                            "stage": TARGET_LABEL,
                            "vpe_group": sign_name,
                            "n_pattern_files": number_of_pattern_files,
                            "n_aligned_trials": number_of_aligned_trials,
                            "output_file": output_path.relative_to(
                                output_root
                            ),
                            "status": "existing",
                            "reason": "",
                        }
                    )

                    continue

                try:
                    require_same_grid(
                        beta_reference,
                        roi_image,
                        "LSS beta reference",
                        f"ROI {roi_name}",
                        arguments.affine_tolerance,
                    )

                    if roi_mask.shape != beta_stack.shape[1:]:
                        raise ValueError(
                            "The ROI mask shape does not match the "
                            f"LSS beta maps: ROI={roi_mask.shape}, "
                            f"beta={beta_stack.shape[1:]}"
                        )

                    # Extract the ROI beta series from the same LSS beta maps
                    # used to calculate the whole-GM nuisance series.
                    roi_trial_series = np.nanmean(
                        beta_stack[:, roi_mask],
                        axis=1,
                    )

                    roi_z = zscore_vector(
                        roi_trial_series
                    )

                    residualized_roi_z = residualize_roi_series(
                        roi_z=roi_z,
                        gm_z=gm_z,
                    )

                    # Pattern-change maps are used as voxel-wise outcomes.
                    (
                        slope_volume,
                        number_of_valid_voxels,
                    ) = fit_voxelwise_slopes(
                        predictor=residualized_roi_z,
                        pattern_stack=pattern_stack,
                    )

                    output_image = create_image_like(
                        pattern_reference,
                        slope_volume,
                    )

                    output_subject_directory.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    nib.save(
                        output_image,
                        str(output_path),
                    )

                    print(
                        f"[Saved] {subject_name}/"
                        f"{roi_name}/{TARGET_LABEL}_{sign_name}"
                    )

                    summary_rows.append(
                        {
                            "roi": roi_name,
                            "participant_id": subject_name,
                            "stage": TARGET_LABEL,
                            "vpe_group": sign_name,
                            "n_pattern_files": number_of_pattern_files,
                            "n_aligned_trials": number_of_aligned_trials,
                            "n_valid_voxels": number_of_valid_voxels,
                            "output_file": output_path.relative_to(
                                output_root
                            ),
                            "status": "completed",
                            "reason": "",
                        }
                    )

                except Exception as error:
                    print(
                        f"[Failed] {subject_name}/"
                        f"{roi_name}/{TARGET_LABEL}_{sign_name}: "
                        f"{error}"
                    )

                    summary_rows.append(
                        {
                            "roi": roi_name,
                            "participant_id": subject_name,
                            "stage": TARGET_LABEL,
                            "vpe_group": sign_name,
                            "n_pattern_files": number_of_pattern_files,
                            "n_aligned_trials": number_of_aligned_trials,
                            "status": "failed",
                            "reason": str(error),
                        }
                    )

    summary_path = write_summary(
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

    print("\n========== Final summary ==========")
    print(f"Completed analyses: {number_completed}")
    print(f"Existing outputs:   {number_existing}")
    print(f"Failed analyses:    {number_failed}")
    print(f"Skipped analyses:   {number_skipped}")
    print(f"Processing report:  {summary_path}")
    print(f"Output root:        {output_root}")


if __name__ == "__main__":
    main()