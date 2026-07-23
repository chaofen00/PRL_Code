#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
First-level ROI-to-pattern-change regression with whole-GM residualization.

For each participant, learning stage (LE/RN), and ROI:

1. Use cue_filtered_trials.csv and the stage-labeled pattern-change files
   to identify and align the relevant trials.

2. For each aligned trial, extract from the outcome-locked LSS beta map:
   - the mean beta value within the ROI;
   - the mean beta value within the whole-GM mask.

3. Within each participant and learning stage:
   - z-score the ROI beta series;
   - z-score the whole-GM beta series;
   - regress the ROI beta series on the whole-GM beta series;
   - z-score the resulting ROI residual series.

4. Use the residualized ROI series as the predictor in a voxel-wise OLS
   regression of the corresponding pattern-change maps.

5. Save the voxel-wise slope map for second-level analysis.

Expected directory structure
----------------------------
project/
├── run_roi_pattern_change_regression.py
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
│   │   │   ├── LE/
│   │   │   │   ├── pattern_change_trial_001.nii
│   │   │   │   └── ...
│   │   │   └── RN/
│   │   └── ...
│   ├── onsets/
│   │   ├── Sub02/
│   │   │   └── csv_output/
│   │   │       └── cue_filtered_trials.csv
│   │   └── ...
│   └── masks/
│       └── group_gm_mask.nii
└── outputs/
    └── roi_pattern_regression/

Dependencies
------------
numpy
nibabel
pandas
nilearn  (optional; used only for new_img_like)

Example
-------
python run_roi_pattern_change_regression.py \
    --roi-dir /path/to/rois \
    --lss-root /path/to/lss_outcome \
    --pattern-root /path/to/pattern_change_labeled \
    --onsets-root /path/to/onsets \
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


DEFAULT_LABELS = ("LE", "RN")

BETA_PREFIX = "beta_outcome_"
PATTERN_PREFIX = "pattern_change_trial_"

NIFTI_SUFFIXES = (".nii", ".nii.gz")

EPSILON = 1e-6


def parse_arguments() -> argparse.Namespace:
    """Parse input paths and analysis settings."""
    script_dir = Path(__file__).resolve().parent
    default_data_root = script_dir / "data"

    parser = argparse.ArgumentParser(
        description=(
            "Run participant-level voxel-wise regressions of pattern change "
            "on whole-GM-residualized ROI beta activity."
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
            "Root directory containing stage-labeled pattern-change maps. "
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
        default=script_dir / "outputs" / "roi_pattern_regression",
        help=(
            "Output directory for participant-level slope maps. "
            "Default: <script_dir>/outputs/roi_pattern_regression"
        ),
    )

    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(DEFAULT_LABELS),
        help="Learning-stage labels to analyze. Default: LE RN",
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
            "Minimum number of aligned trials required for an analysis. "
            "Default: 8"
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
        help="Skip analyses whose slope map already exists.",
    )

    return parser.parse_args()


def natural_sort_key(value: str) -> list[object]:
    """Sort names naturally, for example Sub2 before Sub10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def normalize_subject_id(value: str) -> str:
    """Extract the numeric part of a participant identifier."""
    match = re.search(r"(\d+)", str(value))

    if match is None:
        raise ValueError(
            f"Could not parse participant ID from: {value}"
        )

    return match.group(1)


def normalize_labels(labels: Sequence[str]) -> list[str]:
    """Normalize stage labels and remove duplicate entries."""
    normalized: list[str] = []

    for label in labels:
        clean_label = str(label).strip().upper()

        if clean_label and clean_label not in normalized:
            normalized.append(clean_label)

    if not normalized:
        raise ValueError("At least one stage label is required.")

    return normalized


def strip_nifti_suffix(path: Path) -> str:
    """Return a filename without .nii or .nii.gz."""
    name = path.name

    if name.lower().endswith(".nii.gz"):
        return name[:-7]

    if name.lower().endswith(".nii"):
        return name[:-4]

    return path.stem


def is_nifti_file(path: Path) -> bool:
    """Return True when the path is a supported NIfTI file."""
    return (
        path.is_file()
        and path.name.lower().endswith(NIFTI_SUFFIXES)
    )


def load_3d_image(
    path: Path,
) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    """
    Load a NIfTI image as a three-dimensional float32 array.

    Singleton four-dimensional images are reduced to three dimensions.
    """
    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)

    if data.ndim == 4 and data.shape[-1] == 1:
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
    """Check whether two images have identical dimensions and affines."""
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
    output_data = np.asarray(data, dtype=np.float32)

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

    header = reference_image.header.copy()
    header.set_data_dtype(np.float32)

    return nib.Nifti1Image(
        output_data,
        reference_image.affine,
        header,
    )


def first_existing_nifti(base_path: Path) -> Path | None:
    """Return an existing .nii or .nii.gz version of a path."""
    if base_path.is_file():
        return base_path

    path_text = str(base_path)

    if path_text.lower().endswith(".nii"):
        compressed_path = Path(path_text + ".gz")

        if compressed_path.is_file():
            return compressed_path

    elif not path_text.lower().endswith(".nii.gz"):
        uncompressed_path = Path(path_text + ".nii")
        compressed_path = Path(path_text + ".nii.gz")

        if uncompressed_path.is_file():
            return uncompressed_path

        if compressed_path.is_file():
            return compressed_path

    return None


def trial_from_filename(
    filename: str,
    prefix: str,
) -> int | None:
    """Extract the trial number from a prefixed NIfTI filename."""
    if not filename.lower().startswith(prefix.lower()):
        return None

    remainder = filename[len(prefix):]

    if remainder.lower().endswith(".nii.gz"):
        remainder = remainder[:-7]

    elif remainder.lower().endswith(".nii"):
        remainder = remainder[:-4]

    match = re.search(r"(\d+)", remainder)

    if match is None:
        return None

    return int(match.group(1))


def zscore_vector(
    values: np.ndarray,
    epsilon: float = EPSILON,
) -> np.ndarray:
    """
    Z-score a one-dimensional vector.

    Temporal detrending is not performed.
    """
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError(
            f"Expected a one-dimensional vector, received {values.shape}."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError("The input vector contains NaN or Inf.")

    mean_value = np.mean(values)
    standard_deviation = np.std(values, ddof=0)

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
        key=lambda path: natural_sort_key(path.name),
    )


def find_subject_directory(
    root: Path,
    subject_id: str,
) -> Path | None:
    """Locate a participant directory with common zero-padding formats."""
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

        candidate_path = root / candidate_name

        if candidate_path.is_dir():
            return candidate_path

    return None


def find_lss_directory(
    lss_root: Path,
    subject_name: str,
) -> Path | None:
    """Locate a participant's outcome-locked LSS directory."""
    subject_id = normalize_subject_id(subject_name)
    participant_number = int(subject_id)

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

        checked_names.add(candidate_name)

        candidate_path = lss_root / candidate_name

        if candidate_path.is_dir():
            return candidate_path

    return None


def get_subject_directories(
    pattern_root: Path,
    requested_subject_ids: Sequence[str] | None,
) -> list[Path]:
    """Return requested participant directories or detect all."""
    if requested_subject_ids is None:
        return detect_subject_directories(pattern_root)

    selected: list[Path] = []

    for requested_id in requested_subject_ids:
        subject_id = normalize_subject_id(requested_id)

        subject_directory = find_subject_directory(
            pattern_root,
            subject_id,
        )

        if subject_directory is None:
            print(
                f"[Missing] Participant directory not found: "
                f"{requested_id}"
            )
            continue

        selected.append(subject_directory)

    unique_directories = {
        path.resolve(): path
        for path in selected
    }

    return sorted(
        unique_directories.values(),
        key=lambda path: natural_sort_key(path.name),
    )


def read_trial_label_table(
    csv_path: Path,
    allowed_labels: Sequence[str],
) -> pd.DataFrame:
    """Read and validate trial and label columns from the CSV file."""
    dataframe = None
    decoding_error: Exception | None = None

    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            dataframe = pd.read_csv(
                csv_path,
                encoding=encoding,
            )
            break

        except UnicodeDecodeError as error:
            decoding_error = error

    if dataframe is None:
        raise RuntimeError(
            f"Could not decode CSV file: {csv_path}. "
            f"Last error: {decoding_error}"
        )

    column_lookup = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    if (
        "trial" not in column_lookup
        or "label" not in column_lookup
    ):
        raise ValueError(
            f"CSV file lacks trial or label columns: {csv_path}"
        )

    dataframe = dataframe.rename(
        columns={
            column_lookup["trial"]: "trial",
            column_lookup["label"]: "label",
        }
    )

    dataframe = dataframe[
        ["trial", "label"]
    ].copy()

    dataframe["trial"] = pd.to_numeric(
        dataframe["trial"],
        errors="coerce",
    )

    dataframe["label"] = (
        dataframe["label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    dataframe = dataframe.dropna(
        subset=["trial"]
    ).copy()

    dataframe["trial"] = dataframe["trial"].astype(int)

    dataframe = dataframe.loc[
        dataframe["trial"] >= 1
    ].copy()

    dataframe = dataframe.loc[
        dataframe["label"].isin(allowed_labels)
    ].copy()

    label_counts = (
        dataframe
        .groupby("trial")["label"]
        .nunique()
    )

    conflicting_trials = (
        label_counts[
            label_counts > 1
        ]
        .index
        .tolist()
    )

    if conflicting_trials:
        raise ValueError(
            "The CSV assigns conflicting labels to these trials: "
            f"{conflicting_trials[:20]}"
        )

    dataframe = dataframe.drop_duplicates(
        subset=["trial"],
        keep="first",
    )

    return dataframe.sort_values(
        "trial"
    ).reset_index(drop=True)


def list_pattern_files(
    label_directory: Path,
) -> list[Path]:
    """List stage-specific pattern-change maps sorted by trial number."""
    files_by_trial: dict[int, Path] = {}

    if not label_directory.is_dir():
        return []

    for path in label_directory.iterdir():
        if not is_nifti_file(path):
            continue

        trial_number = trial_from_filename(
            path.name,
            PATTERN_PREFIX,
        )

        if trial_number is None:
            continue

        existing_path = files_by_trial.get(trial_number)

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
        key=lambda path: natural_sort_key(path.name),
    )

    if not roi_files:
        raise FileNotFoundError(
            f"No ROI NIfTI files were found in: {roi_directory}"
        )

    roi_masks = {}

    for roi_path in roi_files:
        roi_image, roi_data = load_3d_image(roi_path)

        roi_mask = (
            np.isfinite(roi_data)
            & (roi_data > 0)
        )

        if not np.any(roi_mask):
            print(
                f"[Warning] Empty ROI mask skipped: {roi_path}"
            )
            continue

        roi_name = strip_nifti_suffix(roi_path)

        if roi_name in roi_masks:
            raise ValueError(
                f"Duplicated ROI name: {roi_name}"
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


def build_aligned_trial_pairs(
    subject_directory: Path,
    lss_directory: Path,
    trial_to_label: dict[int, str],
    label: str,
) -> tuple[list[tuple[int, Path, Path]], int]:
    """
    Match pattern-change maps with outcome-locked LSS beta maps.

    Pattern-change files determine which voxel-wise outcome maps are
    available for each labeled trial.
    """
    label_directory = subject_directory / label

    pattern_files = list_pattern_files(
        label_directory
    )

    aligned_pairs: list[
        tuple[int, Path, Path]
    ] = []

    for pattern_path in pattern_files:
        trial_number = trial_from_filename(
            pattern_path.name,
            PATTERN_PREFIX,
        )

        if trial_number is None:
            continue

        if trial_to_label.get(trial_number) != label:
            continue

        beta_path = first_existing_nifti(
            lss_directory
            / f"{BETA_PREFIX}{trial_number:03d}.nii"
        )

        if beta_path is None:
            print(
                f"[Missing] {subject_directory.name}/{label}, "
                f"trial={trial_number:03d}: "
                "outcome LSS beta was not found."
            )
            continue

        aligned_pairs.append(
            (
                trial_number,
                pattern_path,
                beta_path,
            )
        )

    aligned_pairs.sort(
        key=lambda item: item[0]
    )

    return aligned_pairs, len(pattern_files)


def load_aligned_images(
    aligned_pairs: Sequence[tuple[int, Path, Path]],
    gm_image: nib.spatialimages.SpatialImage,
    affine_tolerance: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    nib.spatialimages.SpatialImage,
    nib.spatialimages.SpatialImage,
]:
    """
    Load aligned beta and pattern-change images.

    Returns
    -------
    beta_stack:
        Outcome-locked LSS beta data with shape T × X × Y × Z.

    pattern_stack:
        Pattern-change data with shape T × X × Y × Z.

    beta_reference:
        Reference image for beta-space mask validation.

    pattern_reference:
        Reference image used to save the slope map.
    """
    beta_arrays: list[np.ndarray] = []
    pattern_arrays: list[np.ndarray] = []

    beta_reference = None
    beta_reference_path = None

    pattern_reference = None
    pattern_reference_path = None

    for _, pattern_path, beta_path in aligned_pairs:
        beta_image, beta_data = load_3d_image(beta_path)
        pattern_image, pattern_data = load_3d_image(pattern_path)

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

        beta_arrays.append(beta_data)
        pattern_arrays.append(pattern_data)

    if (
        beta_reference is None
        or pattern_reference is None
    ):
        raise ValueError(
            "No aligned beta and pattern-change images were loaded."
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
    design = np.column_stack(
        (
            gm_z,
            np.ones_like(gm_z),
        )
    )

    coefficients, _, rank, _ = np.linalg.lstsq(
        design,
        roi_z,
        rcond=None,
    )

    if rank < 2:
        raise ValueError(
            "The whole-GM residualization model is rank deficient."
        )

    residuals = (
        roi_z
        - design @ coefficients
    )

    return zscore_vector(residuals)


def fit_voxelwise_slopes(
    predictor: np.ndarray,
    pattern_stack: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Fit the voxel-wise model:

        pattern_change = intercept + slope * predictor

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
            "No voxel has finite values across all included trials."
        )

    valid_data = voxel_matrix[
        :,
        valid_voxels,
    ]

    design = np.column_stack(
        (
            predictor,
            np.ones_like(predictor),
        )
    )

    coefficients, _, rank, _ = np.linalg.lstsq(
        design,
        valid_data,
        rcond=None,
    )

    if rank < 2:
        raise ValueError(
            "The voxel-wise regression design is rank deficient."
        )

    slope_values = coefficients[0, :]

    slope_flat = np.full(
        voxel_matrix.shape[1],
        np.nan,
        dtype=np.float32,
    )

    slope_flat[valid_voxels] = slope_values.astype(
        np.float32
    )

    slope_volume = slope_flat.reshape(
        pattern_stack.shape[1:]
    )

    return slope_volume, number_of_valid_voxels


def clean_tsv_field(value: object) -> str:
    """Remove tabs and line breaks before writing TSV output."""
    return (
        str(value)
        .replace("\t", " ")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def write_summary(
    output_root: Path,
    rows: list[dict[str, object]],
) -> Path:
    """Write a processing report for all ROI-level analyses."""
    summary_path = (
        output_root
        / "first_level_regression_summary.tsv"
    )

    columns = [
        "roi",
        "participant_id",
        "label",
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

        for row in rows:
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
    """Run the participant-level regression workflow."""
    args = parse_arguments()

    roi_directory = args.roi_dir.expanduser().resolve()
    lss_root = args.lss_root.expanduser().resolve()
    pattern_root = args.pattern_root.expanduser().resolve()
    onsets_root = args.onsets_root.expanduser().resolve()
    gm_mask_path = args.gm_mask.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    labels = normalize_labels(args.labels)

    required_directories = {
        "ROI directory": roi_directory,
        "LSS root": lss_root,
        "Pattern-change root": pattern_root,
        "Onsets root": onsets_root,
    }

    for description, path in required_directories.items():
        if not path.is_dir():
            raise NotADirectoryError(
                f"{description} does not exist: {path}"
            )

    if not gm_mask_path.is_file():
        raise FileNotFoundError(
            f"Whole-GM mask does not exist: {gm_mask_path}"
        )

    if args.minimum_trials < 3:
        raise ValueError(
            "--minimum-trials must be at least 3."
        )

    if args.affine_tolerance < 0:
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

    if not np.any(gm_mask):
        raise ValueError(
            f"The whole-GM mask is empty: {gm_mask_path}"
        )

    subject_directories = get_subject_directories(
        pattern_root,
        args.subject_ids,
    )

    if not subject_directories:
        raise RuntimeError(
            "No participant directories were found."
        )

    print(f"ROI directory:        {roi_directory}")
    print(f"LSS root:             {lss_root}")
    print(f"Pattern-change root:  {pattern_root}")
    print(f"Onsets root:          {onsets_root}")
    print(f"Whole-GM mask:        {gm_mask_path}")
    print(f"Output root:          {output_root}")
    print(f"ROIs:                 {len(roi_masks)}")
    print(f"Participants:         {len(subject_directories)}")
    print(f"Labels:               {', '.join(labels)}")
    print(f"Minimum trials:       {args.minimum_trials}")
    print(f"Skip existing:        {args.skip_existing}")

    summary_rows: list[dict[str, object]] = []

    for subject_directory in subject_directories:
        subject_name = subject_directory.name
        subject_id = normalize_subject_id(subject_name)

        lss_directory = find_lss_directory(
            lss_root,
            subject_name,
        )

        onset_subject_directory = find_subject_directory(
            onsets_root,
            subject_id,
        )

        if (
            lss_directory is None
            or onset_subject_directory is None
        ):
            reason = (
                "LSS or onset participant directory was not found"
            )

            print(
                f"[Skipped] {subject_name}: {reason}."
            )

            for label in labels:
                for roi_name in roi_masks:
                    summary_rows.append(
                        {
                            "roi": roi_name,
                            "participant_id": subject_name,
                            "label": label,
                            "status": "skipped",
                            "reason": reason,
                        }
                    )

            continue

        csv_path = (
            onset_subject_directory
            / "csv_output"
            / "cue_filtered_trials.csv"
        )

        if not csv_path.is_file():
            reason = "cue_filtered_trials.csv was not found"

            print(
                f"[Skipped] {subject_name}: {reason}."
            )

            for label in labels:
                for roi_name in roi_masks:
                    summary_rows.append(
                        {
                            "roi": roi_name,
                            "participant_id": subject_name,
                            "label": label,
                            "status": "skipped",
                            "reason": reason,
                        }
                    )

            continue

        try:
            trial_label_table = read_trial_label_table(
                csv_path,
                labels,
            )

        except Exception as error:
            print(
                f"[Skipped] {subject_name}: {error}"
            )

            for label in labels:
                for roi_name in roi_masks:
                    summary_rows.append(
                        {
                            "roi": roi_name,
                            "participant_id": subject_name,
                            "label": label,
                            "status": "skipped",
                            "reason": str(error),
                        }
                    )

            continue

        trial_to_label = dict(
            zip(
                trial_label_table["trial"],
                trial_label_table["label"],
            )
        )

        for label in labels:
            (
                aligned_pairs,
                number_of_pattern_files,
            ) = build_aligned_trial_pairs(
                subject_directory=subject_directory,
                lss_directory=lss_directory,
                trial_to_label=trial_to_label,
                label=label,
            )

            number_of_aligned_trials = len(
                aligned_pairs
            )

            if number_of_aligned_trials < args.minimum_trials:
                reason = (
                    f"too few aligned trials "
                    f"({number_of_aligned_trials} < "
                    f"{args.minimum_trials})"
                )

                print(
                    f"[Skipped] {subject_name}/{label}: {reason}."
                )

                for roi_name in roi_masks:
                    summary_rows.append(
                        {
                            "roi": roi_name,
                            "participant_id": subject_name,
                            "label": label,
                            "n_pattern_files": number_of_pattern_files,
                            "n_aligned_trials": number_of_aligned_trials,
                            "status": "skipped",
                            "reason": reason,
                        }
                    )

                continue

            try:
                (
                    beta_stack,
                    pattern_stack,
                    beta_reference,
                    pattern_reference,
                ) = load_aligned_images(
                    aligned_pairs=aligned_pairs,
                    gm_image=gm_image,
                    affine_tolerance=args.affine_tolerance,
                )

                if gm_mask.shape != beta_stack.shape[1:]:
                    raise ValueError(
                        "The whole-GM mask shape does not match the "
                        f"LSS beta images: mask={gm_mask.shape}, "
                        f"beta={beta_stack.shape[1:]}"
                    )

                # Extract the whole-GM nuisance series from the
                # outcome-locked LSS beta maps.
                gm_trial_series = np.nanmean(
                    beta_stack[:, gm_mask],
                    axis=1,
                )

                gm_z = zscore_vector(
                    gm_trial_series
                )

            except Exception as error:
                print(
                    f"[Failed] {subject_name}/{label}: {error}"
                )

                for roi_name in roi_masks:
                    summary_rows.append(
                        {
                            "roi": roi_name,
                            "participant_id": subject_name,
                            "label": label,
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
                        f"regBeta_GMresid_{label}"
                        f"{args.output_extension}"
                    )
                )

                if (
                    args.skip_existing
                    and output_path.is_file()
                ):
                    print(
                        f"[Existing] "
                        f"{subject_name}/{roi_name}/{label}"
                    )

                    summary_rows.append(
                        {
                            "roi": roi_name,
                            "participant_id": subject_name,
                            "label": label,
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
                        args.affine_tolerance,
                    )

                    if roi_mask.shape != beta_stack.shape[1:]:
                        raise ValueError(
                            "The ROI mask shape does not match the "
                            f"LSS beta images: ROI={roi_mask.shape}, "
                            f"beta={beta_stack.shape[1:]}"
                        )

                    # Extract the ROI series from the same outcome-locked
                    # LSS beta maps used for the whole-GM series.
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

                    # Pattern-change maps are the voxel-wise outcomes.
                    slope_volume, number_of_valid_voxels = (
                        fit_voxelwise_slopes(
                            predictor=residualized_roi_z,
                            pattern_stack=pattern_stack,
                        )
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
                        f"[Saved] "
                        f"{subject_name}/{roi_name}/{label}"
                    )

                    summary_rows.append(
                        {
                            "roi": roi_name,
                            "participant_id": subject_name,
                            "label": label,
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
                        f"[Failed] "
                        f"{subject_name}/{roi_name}/{label}: "
                        f"{error}"
                    )

                    summary_rows.append(
                        {
                            "roi": roi_name,
                            "participant_id": subject_name,
                            "label": label,
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