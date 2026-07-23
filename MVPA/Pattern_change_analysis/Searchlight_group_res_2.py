#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute trial-to-trial multivoxel pattern change using a searchlight.

For each participant, cue-locked LSS beta images from adjacent trials are
compared within a spherical searchlight. Pattern change is defined as:

    pattern_change[t, t+1]
        = -arctanh(corr(pattern[t], pattern[t+1]))

where the correlation is calculated across valid gray-matter voxels inside
the searchlight.

Public-release version
----------------------
- Contains no local drive letters, user names, or machine-specific paths.
- Input, output, and group-mask paths are provided through command-line
  arguments.
- Participants can be detected automatically.
- Both .nii and .nii.gz beta images are supported.
- Image dimensions and affine matrices are checked before analysis.
- Existing valid output maps can be skipped for resume support.
- Participant-level processing results are written to a TSV summary.

Expected input structure
------------------------
<input_root>/
├── Sub02_LSS_cue/
│   ├── beta_cue_001.nii
│   ├── beta_cue_002.nii
│   └── ...
├── Sub03_LSS_cue/
│   └── ...
└── ...

Output structure
----------------
<output_root>/
├── Sub02_Pattern_change/
│   ├── pattern_change_trial_001.nii
│   ├── pattern_change_trial_002.nii
│   ├── center_coverage_map.nii
│   └── processing_summary.txt
├── Sub03_Pattern_change/
│   └── ...
└── pattern_change_processing_summary.tsv

Dependencies
------------
pip install numpy nibabel nilearn joblib

Example
-------
python compute_searchlight_pattern_change.py \
    --input-root /path/to/lss_cue \
    --output-root /path/to/pattern_change \
    --group-mask /path/to/group_gm_mask.nii.gz \
    --searchlight-radius 4 \
    --min-voxels 20 \
    --n-jobs 12

Process selected participants only
----------------------------------
python compute_searchlight_pattern_change.py \
    --subject-ids 02 03 04 05

Resume an interrupted analysis
------------------------------
python compute_searchlight_pattern_change.py \
    --skip-existing
"""

from __future__ import annotations

import argparse
import multiprocessing
import re
import time
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from nilearn.image import load_img, new_img_like


BETA_FILE_PATTERN = re.compile(
    r"^beta_cue_(\d+)\.nii(?:\.gz)?$",
    flags=re.IGNORECASE,
)

SUBJECT_DIRECTORY_PATTERN = re.compile(
    r"^Sub(\d+)_LSS_cue$",
    flags=re.IGNORECASE,
)


def parse_arguments() -> argparse.Namespace:
    """Parse portable input, output, and searchlight settings."""
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Compute adjacent-trial multivoxel pattern change using "
            "a spherical searchlight."
        )
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=script_dir / "data" / "lss_cue",
        help=(
            "Root directory containing SubXX_LSS_cue folders. "
            "Default: <script_dir>/data/lss_cue"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=script_dir / "outputs" / "pattern_change",
        help=(
            "Root directory for trial-wise pattern-change maps. "
            "Default: <script_dir>/outputs/pattern_change"
        ),
    )

    parser.add_argument(
        "--group-mask",
        type=Path,
        default=script_dir / "data" / "masks" / "group_gm_mask.nii.gz",
        help=(
            "Three-dimensional group gray-matter mask. "
            "Default: <script_dir>/data/masks/group_gm_mask.nii.gz"
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
        "--searchlight-radius",
        type=int,
        default=4,
        help=(
            "Searchlight radius measured in voxels. "
            "Default: 4"
        ),
    )

    parser.add_argument(
        "--min-voxels",
        type=int,
        default=20,
        help=(
            "Minimum number of valid gray-matter voxels required inside "
            "a searchlight. Default: 20"
        ),
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=min(
            multiprocessing.cpu_count(),
            12,
        ),
        help=(
            "Number of parallel workers. "
            "Default: min(number of CPU cores, 12)"
        ),
    )

    parser.add_argument(
        "--debug-level",
        type=int,
        choices=(0, 1, 2, 3),
        default=1,
        help=(
            "Logging detail: 0=quiet, 1=summary, 2=trial progress, "
            "3=voxel details. Default: 1"
        ),
    )

    parser.add_argument(
        "--affine-tolerance",
        type=float,
        default=1e-5,
        help=(
            "Absolute tolerance used to compare NIfTI affine matrices. "
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
        help=(
            "Skip an adjacent-trial comparison when its existing output "
            "map can be read successfully."
        ),
    )

    return parser.parse_args()


def debug(
    current_level: int,
    required_level: int,
    *message: object,
) -> None:
    """Print a message when the requested debug level is enabled."""
    if current_level >= required_level:
        print(*message)


def natural_sort_key(value: str) -> list[object]:
    """Sort text naturally, for example Sub2 before Sub10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def normalize_subject_id(value: str) -> str:
    """
    Extract the numeric component of a participant identifier.

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
            f"Could not parse participant ID from: {value}"
        )

    return match.group(1)


def detect_subject_directories(
    input_root: Path,
) -> list[Path]:
    """Detect participant directories matching Sub<number>_LSS_cue."""
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
    Locate a participant LSS directory using common zero-padding variants.

    Supported examples
    ------------------
    Sub2_LSS_cue
    Sub02_LSS_cue
    Sub002_LSS_cue
    """
    participant_number = int(subject_id)

    candidate_names = [
        f"Sub{subject_id}_LSS_cue",
        f"Sub{participant_number}_LSS_cue",
        f"Sub{participant_number:02d}_LSS_cue",
        f"Sub{participant_number:03d}_LSS_cue",
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
    """Return requested participant directories or detect all."""
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
                f"[Missing] LSS directory not found for "
                f"participant {requested_id}."
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


def extract_subject_id(
    subject_directory: Path,
) -> str:
    """Extract the numeric participant ID from an LSS folder name."""
    match = SUBJECT_DIRECTORY_PATTERN.fullmatch(
        subject_directory.name
    )

    if match is None:
        raise ValueError(
            "Unexpected participant directory name: "
            f"{subject_directory.name}"
        )

    return match.group(1)


def list_beta_files(
    input_directory: Path,
) -> list[Path]:
    """
    Return cue beta images sorted by trial number.

    If both .nii and .nii.gz versions exist for the same trial, .nii.gz is
    preferred so that a trial is not counted twice.
    """
    files_by_trial: dict[int, Path] = {}

    for path in input_directory.iterdir():
        if not path.is_file():
            continue

        match = BETA_FILE_PATTERN.fullmatch(
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


def load_3d_data(
    image_path: Path,
) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    """Load a 3D NIfTI or reduce a singleton 4D NIfTI to 3D."""
    image = load_img(
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
            "Expected a 3D image or singleton 4D image, "
            f"but received shape {data.shape}: {image_path}"
        )

    return image, data


def same_grid(
    reference_image: nib.spatialimages.SpatialImage,
    current_image: nib.spatialimages.SpatialImage,
    affine_tolerance: float,
) -> bool:
    """Check whether two NIfTI images share dimensions and affine."""
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
    """Raise an error if two images do not share a common grid."""
    if same_grid(
        reference_image,
        current_image,
        affine_tolerance,
    ):
        return

    raise ValueError(
        "NIfTI grid mismatch.\n"
        f"Reference: {reference_name}\n"
        f"Reference shape: {reference_image.shape}\n"
        f"Current: {current_name}\n"
        f"Current shape: {current_image.shape}"
    )


def create_searchlight_offsets(
    radius: int,
) -> np.ndarray:
    """
    Create integer offsets for a spherical Euclidean searchlight.

    Radius is expressed in voxel units.
    """
    axis = np.arange(
        -radius,
        radius + 1,
    )

    x_grid, y_grid, z_grid = np.meshgrid(
        axis,
        axis,
        axis,
        indexing="ij",
    )

    sphere = (
        x_grid**2
        + y_grid**2
        + z_grid**2
    ) <= radius**2

    return (
        np.argwhere(sphere)
        - np.array(
            [
                radius,
                radius,
                radius,
            ]
        )
    )


def is_valid_nifti(
    image_path: Path,
) -> bool:
    """Return True when a NIfTI exists and can be read as a 3D image."""
    if not image_path.is_file():
        return False

    try:
        image = nib.load(
            str(image_path)
        )

        shape = image.shape

        return (
            len(shape) == 3
            or (
                len(shape) == 4
                and shape[-1] == 1
            )
        )

    except Exception:
        return False


def save_image_like(
    template_image: nib.spatialimages.SpatialImage,
    data: np.ndarray,
    output_path: Path,
    dtype: np.dtype,
) -> None:
    """Save an array using the template image's grid and header."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_image = new_img_like(
        template_image,
        np.asarray(
            data,
            dtype=dtype,
        ),
        copy_header=True,
    )

    output_image.to_filename(
        str(output_path)
    )


def compute_pattern_change_for_trial(
    trial_index: int,
    *,
    data: np.ndarray,
    group_mask: np.ndarray,
    mask_coordinates: np.ndarray,
    searchlight_offsets: np.ndarray,
    minimum_voxels: int,
    template_image: nib.spatialimages.SpatialImage,
    output_directory: Path,
    output_extension: str,
    skip_existing: bool,
    debug_level: int,
) -> tuple[int, int, list[tuple[int, int, int]], str]:
    """
    Compute one pattern-change map between trial t and trial t+1.

    Returns
    -------
    trial_index:
        Zero-based trial-pair index.
    valid_centers:
        Number of searchlight centers written.
    written_coordinates:
        Coordinates with valid pattern-change estimates.
    status:
        completed or existing.
    """
    number_of_trials = data.shape[0]
    spatial_shape = data.shape[1:]

    output_path = (
        output_directory
        / (
            f"pattern_change_trial_"
            f"{trial_index + 1:03d}"
            f"{output_extension}"
        )
    )

    if (
        skip_existing
        and is_valid_nifti(output_path)
    ):
        existing_data = nib.load(
            str(output_path)
        ).get_fdata(
            dtype=np.float32
        )

        written_coordinates_array = np.argwhere(
            np.isfinite(existing_data)
        )

        written_coordinates = [
            tuple(
                int(value)
                for value in coordinate
            )
            for coordinate in written_coordinates_array
        ]

        return (
            trial_index,
            len(written_coordinates),
            written_coordinates,
            "existing",
        )

    debug(
        debug_level,
        2,
        (
            f"Processing trial pair "
            f"{trial_index + 1}/{number_of_trials - 1}"
        ),
    )

    pattern_change_map = np.full(
        spatial_shape,
        np.nan,
        dtype=np.float32,
    )

    valid_centers = 0
    written_coordinates: list[tuple[int, int, int]] = []

    x_size, y_size, z_size = spatial_shape

    first_trial_volume = data[
        trial_index
    ]

    second_trial_volume = data[
        trial_index + 1
    ]

    for center_coordinate in mask_coordinates:
        center_x, center_y, center_z = (
            int(center_coordinate[0]),
            int(center_coordinate[1]),
            int(center_coordinate[2]),
        )

        neighborhood_coordinates = (
            searchlight_offsets
            + np.array(
                [
                    center_x,
                    center_y,
                    center_z,
                ]
            )
        )

        inside_bounds = (
            (neighborhood_coordinates[:, 0] >= 0)
            & (neighborhood_coordinates[:, 0] < x_size)
            & (neighborhood_coordinates[:, 1] >= 0)
            & (neighborhood_coordinates[:, 1] < y_size)
            & (neighborhood_coordinates[:, 2] >= 0)
            & (neighborhood_coordinates[:, 2] < z_size)
        )

        neighborhood_coordinates = (
            neighborhood_coordinates[
                inside_bounds
            ]
        )

        if neighborhood_coordinates.shape[0] < minimum_voxels:
            continue

        neighborhood_in_mask = group_mask[
            neighborhood_coordinates[:, 0],
            neighborhood_coordinates[:, 1],
            neighborhood_coordinates[:, 2],
        ]

        neighborhood_coordinates = (
            neighborhood_coordinates[
                neighborhood_in_mask
            ]
        )

        if neighborhood_coordinates.shape[0] < minimum_voxels:
            continue

        pattern_first = first_trial_volume[
            neighborhood_coordinates[:, 0],
            neighborhood_coordinates[:, 1],
            neighborhood_coordinates[:, 2],
        ]

        pattern_second = second_trial_volume[
            neighborhood_coordinates[:, 0],
            neighborhood_coordinates[:, 1],
            neighborhood_coordinates[:, 2],
        ]

        finite_values = (
            np.isfinite(pattern_first)
            & np.isfinite(pattern_second)
        )

        if int(finite_values.sum()) < minimum_voxels:
            continue

        valid_pattern_first = pattern_first[
            finite_values
        ]

        valid_pattern_second = pattern_second[
            finite_values
        ]

        # Pearson correlation is undefined when either pattern is constant.
        if (
            np.std(valid_pattern_first) == 0
            or np.std(valid_pattern_second) == 0
        ):
            continue

        correlation = np.corrcoef(
            valid_pattern_first,
            valid_pattern_second,
        )[0, 1]

        if not np.isfinite(correlation):
            continue

        clipped_correlation = np.clip(
            correlation,
            -0.999,
            0.999,
        )

        pattern_change_value = -np.arctanh(
            clipped_correlation
        )

        pattern_change_map[
            center_x,
            center_y,
            center_z,
        ] = pattern_change_value

        valid_centers += 1

        written_coordinates.append(
            (
                center_x,
                center_y,
                center_z,
            )
        )

        if debug_level >= 3:
            print(
                f"  voxel ({center_x}, {center_y}, {center_z}) "
                f"r={correlation:.3f}, "
                f"pattern_change={pattern_change_value:.3f}"
            )

    number_of_mask_voxels = int(
        group_mask.sum()
    )

    coverage_percent = (
        valid_centers
        / number_of_mask_voxels
        * 100.0
        if number_of_mask_voxels > 0
        else 0.0
    )

    debug(
        debug_level,
        1,
        (
            f"Trial pair {trial_index + 1}: "
            f"valid centers={valid_centers}, "
            f"mask coverage={coverage_percent:.1f}%"
        ),
    )

    save_image_like(
        template_image=template_image,
        data=pattern_change_map,
        output_path=output_path,
        dtype=np.float32,
    )

    return (
        trial_index,
        valid_centers,
        written_coordinates,
        "completed",
    )


def run_subject(
    subject_directory: Path,
    *,
    output_root: Path,
    group_mask_image: nib.spatialimages.SpatialImage,
    group_mask: np.ndarray,
    searchlight_radius: int,
    minimum_voxels: int,
    number_of_jobs: int,
    debug_level: int,
    affine_tolerance: float,
    output_extension: str,
    skip_existing: bool,
) -> dict[str, object]:
    """Process all adjacent trial pairs for one participant."""
    subject_id = extract_subject_id(
        subject_directory
    )

    output_directory = (
        output_root
        / f"Sub{subject_id}_Pattern_change"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    beta_files = list_beta_files(
        subject_directory
    )

    number_of_trials = len(
        beta_files
    )

    if number_of_trials < 2:
        raise ValueError(
            f"Sub{subject_id} requires at least two cue beta images; "
            f"found {number_of_trials}."
        )

    debug(
        debug_level,
        1,
        (
            f"\n===== Processing Sub{subject_id}: "
            f"{number_of_trials} trial beta images ====="
        ),
    )

    loading_start = time.time()

    images: list[nib.spatialimages.SpatialImage] = []
    trial_data: list[np.ndarray] = []

    reference_image = None
    reference_path = None

    for beta_path in beta_files:
        image, image_data = load_3d_data(
            beta_path
        )

        if reference_image is None:
            reference_image = image
            reference_path = beta_path
        else:
            require_same_grid(
                reference_image,
                image,
                str(reference_path),
                str(beta_path),
                affine_tolerance,
            )

        images.append(image)
        trial_data.append(image_data)

    require_same_grid(
        reference_image,
        group_mask_image,
        str(reference_path),
        "group mask",
        affine_tolerance,
    )

    data = np.stack(
        trial_data,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    spatial_shape = data.shape[1:]

    debug(
        debug_level,
        1,
        (
            f"Data shape [trial, X, Y, Z] = {data.shape}; "
            f"loading time={time.time() - loading_start:.1f} s"
        ),
    )

    number_of_mask_voxels = int(
        group_mask.sum()
    )

    debug(
        debug_level,
        1,
        (
            f"Gray-matter mask voxels: "
            f"{number_of_mask_voxels}"
        ),
    )

    searchlight_offsets = create_searchlight_offsets(
        searchlight_radius
    )

    debug(
        debug_level,
        1,
        (
            f"Searchlight radius={searchlight_radius} voxels; "
            f"full sphere size={len(searchlight_offsets)} voxels"
        ),
    )

    if minimum_voxels > len(searchlight_offsets):
        raise ValueError(
            f"Minimum voxels ({minimum_voxels}) exceeds the full "
            f"searchlight sphere size ({len(searchlight_offsets)})."
        )

    mask_coordinates = np.argwhere(
        group_mask
    )

    coverage_map = np.zeros(
        spatial_shape,
        dtype=np.uint32,
    )

    debug(
        debug_level,
        1,
        (
            f"Using {number_of_jobs} parallel workers for "
            f"{number_of_trials - 1} adjacent trial pairs."
        ),
    )

    processing_start = time.time()

    results = Parallel(
        n_jobs=number_of_jobs,
        backend="loky",
    )(
        delayed(
            compute_pattern_change_for_trial
        )(
            trial_index,
            data=data,
            group_mask=group_mask,
            mask_coordinates=mask_coordinates,
            searchlight_offsets=searchlight_offsets,
            minimum_voxels=minimum_voxels,
            template_image=reference_image,
            output_directory=output_directory,
            output_extension=output_extension,
            skip_existing=skip_existing,
            debug_level=debug_level,
        )
        for trial_index in range(
            number_of_trials - 1
        )
    )

    completed_pairs = 0
    existing_pairs = 0

    for (
        trial_index,
        valid_center_count,
        coordinates,
        status,
    ) in results:
        debug(
            debug_level,
            2,
            (
                f"Trial pair {trial_index + 1}: "
                f"{valid_center_count} centers written; "
                f"status={status}"
            ),
        )

        if status == "existing":
            existing_pairs += 1
        else:
            completed_pairs += 1

        for x_coordinate, y_coordinate, z_coordinate in coordinates:
            coverage_map[
                x_coordinate,
                y_coordinate,
                z_coordinate,
            ] += 1

    coverage_path = (
        output_directory
        / (
            "center_coverage_map"
            f"{output_extension}"
        )
    )

    save_image_like(
        template_image=reference_image,
        data=coverage_map.astype(np.float32),
        output_path=coverage_path,
        dtype=np.float32,
    )

    covered_voxels = int(
        np.count_nonzero(
            coverage_map > 0
        )
    )

    total_spatial_voxels = int(
        np.prod(
            spatial_shape
        )
    )

    whole_volume_coverage_percent = (
        covered_voxels
        / total_spatial_voxels
        * 100.0
        if total_spatial_voxels > 0
        else 0.0
    )

    mask_coverage_percent = (
        covered_voxels
        / number_of_mask_voxels
        * 100.0
        if number_of_mask_voxels > 0
        else 0.0
    )

    positive_coverage = coverage_map[
        coverage_map > 0
    ]

    minimum_coverage = (
        int(positive_coverage.min())
        if positive_coverage.size > 0
        else 0
    )

    maximum_coverage = int(
        coverage_map.max()
    )

    processing_seconds = (
        time.time()
        - processing_start
    )

    participant_summary_path = (
        output_directory
        / "processing_summary.txt"
    )

    with participant_summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            f"participant_id\tSub{subject_id}\n"
        )
        file.write(
            f"n_trial_beta_images\t{number_of_trials}\n"
        )
        file.write(
            f"n_adjacent_trial_pairs\t{number_of_trials - 1}\n"
        )
        file.write(
            f"n_newly_computed_pairs\t{completed_pairs}\n"
        )
        file.write(
            f"n_existing_pairs\t{existing_pairs}\n"
        )
        file.write(
            f"searchlight_radius_voxels\t{searchlight_radius}\n"
        )
        file.write(
            f"minimum_valid_voxels\t{minimum_voxels}\n"
        )
        file.write(
            f"group_mask_voxels\t{number_of_mask_voxels}\n"
        )
        file.write(
            f"covered_center_voxels\t{covered_voxels}\n"
        )
        file.write(
            f"mask_coverage_percent\t{mask_coverage_percent:.6f}\n"
        )
        file.write(
            f"whole_volume_coverage_percent\t"
            f"{whole_volume_coverage_percent:.6f}\n"
        )
        file.write(
            f"minimum_center_coverage_count\t{minimum_coverage}\n"
        )
        file.write(
            f"maximum_center_coverage_count\t{maximum_coverage}\n"
        )
        file.write(
            f"processing_seconds\t{processing_seconds:.3f}\n"
        )
        file.write(
            f"coverage_map\t{coverage_path}\n"
        )

    print(
        f"\n========== Sub{subject_id} summary =========="
    )
    print(
        f"Trial beta images:          {number_of_trials}"
    )
    print(
        f"Adjacent trial pairs:       {number_of_trials - 1}"
    )
    print(
        f"Newly computed pairs:       {completed_pairs}"
    )
    print(
        f"Existing pairs reused:      {existing_pairs}"
    )
    print(
        f"Valid center voxels:        {covered_voxels}"
    )
    print(
        f"Coverage within GM mask:    {mask_coverage_percent:.1f}%"
    )
    print(
        f"Coverage of full volume:    "
        f"{whole_volume_coverage_percent:.1f}%"
    )
    print(
        f"Minimum coverage count:     {minimum_coverage}"
    )
    print(
        f"Maximum coverage count:     {maximum_coverage}"
    )
    print(
        f"Coverage map:               {coverage_path}"
    )
    print(
        f"Processing time:            {processing_seconds:.1f} s"
    )

    return {
        "participant_id": f"Sub{subject_id}",
        "n_trial_beta_images": number_of_trials,
        "n_adjacent_trial_pairs": number_of_trials - 1,
        "n_newly_computed_pairs": completed_pairs,
        "n_existing_pairs": existing_pairs,
        "searchlight_radius_voxels": searchlight_radius,
        "minimum_valid_voxels": minimum_voxels,
        "group_mask_voxels": number_of_mask_voxels,
        "covered_center_voxels": covered_voxels,
        "mask_coverage_percent": mask_coverage_percent,
        "whole_volume_coverage_percent": (
            whole_volume_coverage_percent
        ),
        "minimum_center_coverage_count": minimum_coverage,
        "maximum_center_coverage_count": maximum_coverage,
        "processing_seconds": processing_seconds,
        "status": "completed",
        "error": "",
    }


def clean_tsv_field(value: object) -> str:
    """Remove tabs and line breaks before writing a TSV field."""
    return (
        str(value)
        .replace("\t", " ")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def write_processing_summary(
    output_root: Path,
    summary_rows: list[dict[str, object]],
) -> Path:
    """Write the batch-level participant processing summary."""
    summary_path = (
        output_root
        / "pattern_change_processing_summary.tsv"
    )

    columns = [
        "participant_id",
        "n_trial_beta_images",
        "n_adjacent_trial_pairs",
        "n_newly_computed_pairs",
        "n_existing_pairs",
        "searchlight_radius_voxels",
        "minimum_valid_voxels",
        "group_mask_voxels",
        "covered_center_voxels",
        "mask_coverage_percent",
        "whole_volume_coverage_percent",
        "minimum_center_coverage_count",
        "maximum_center_coverage_count",
        "processing_seconds",
        "status",
        "error",
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
            file.write(
                "\t".join(
                    clean_tsv_field(
                        row.get(column, "")
                    )
                    for column in columns
                )
                + "\n"
            )

    return summary_path


def main() -> None:
    """Run searchlight pattern-change analysis for all selected participants."""
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

    group_mask_path = (
        args.group_mask
        .expanduser()
        .resolve()
    )

    if not input_root.is_dir():
        raise NotADirectoryError(
            "LSS cue input directory does not exist: "
            f"{input_root}"
        )

    if not group_mask_path.is_file():
        raise FileNotFoundError(
            "Group gray-matter mask does not exist: "
            f"{group_mask_path}"
        )

    if args.searchlight_radius < 1:
        raise ValueError(
            "--searchlight-radius must be at least 1."
        )

    if args.min_voxels < 2:
        raise ValueError(
            "--min-voxels must be at least 2."
        )

    if args.n_jobs < 1:
        raise ValueError(
            "--n-jobs must be at least 1."
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
            "No participant LSS directories were found or selected."
        )

    group_mask_image, group_mask_data = load_3d_data(
        group_mask_path
    )

    group_mask = (
        np.isfinite(group_mask_data)
        & (group_mask_data > 0)
    )

    if not np.any(group_mask):
        raise ValueError(
            "The group gray-matter mask contains no positive voxels."
        )

    print(
        f"Input root:              {input_root}"
    )
    print(
        f"Output root:             {output_root}"
    )
    print(
        f"Group mask:              {group_mask_path}"
    )
    print(
        f"Participants:            {len(subject_directories)}"
    )
    print(
        f"Searchlight radius:      {args.searchlight_radius} voxels"
    )
    print(
        f"Minimum valid voxels:    {args.min_voxels}"
    )
    print(
        f"Parallel workers:        {args.n_jobs}"
    )
    print(
        f"Resume existing outputs: {args.skip_existing}"
    )

    summary_rows: list[dict[str, object]] = []

    for subject_directory in subject_directories:
        subject_id = extract_subject_id(
            subject_directory
        )

        try:
            summary = run_subject(
                subject_directory,
                output_root=output_root,
                group_mask_image=group_mask_image,
                group_mask=group_mask,
                searchlight_radius=args.searchlight_radius,
                minimum_voxels=args.min_voxels,
                number_of_jobs=args.n_jobs,
                debug_level=args.debug_level,
                affine_tolerance=args.affine_tolerance,
                output_extension=args.output_extension,
                skip_existing=args.skip_existing,
            )

            summary_rows.append(
                summary
            )

        except Exception as error:
            print(
                f"[Failed] Sub{subject_id}: {error}"
            )

            summary_rows.append(
                {
                    "participant_id": f"Sub{subject_id}",
                    "status": "failed",
                    "error": str(error),
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

    number_failed = sum(
        row.get("status") == "failed"
        for row in summary_rows
    )

    print("\n========== Batch summary ==========")
    print(
        f"Participants completed: {number_completed}"
    )
    print(
        f"Participants failed:    {number_failed}"
    )
    print(
        f"Processing summary:     {summary_path}"
    )
    print(
        f"Output root:            {output_root}"
    )


if __name__ == "__main__":
    main()