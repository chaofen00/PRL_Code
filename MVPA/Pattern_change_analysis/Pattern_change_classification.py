#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classify trial-wise pattern-change images according to LN/LE/RN/RE labels.

For each participant, this script reads:

    <onsets_root>/SubXX/csv_output/cue_filtered_trials.csv

It then copies the corresponding trial-wise pattern-change images from:

    <pattern_root>/SubXX_Pattern_change/
        pattern_change_trial_001.nii
        pattern_change_trial_002.nii
        ...

to:

    <output_root>/SubXX/LN/
    <output_root>/SubXX/LE/
    <output_root>/SubXX/RN/
    <output_root>/SubXX/RE/

Public-release version
----------------------
- Contains no user-specific drive letters or absolute paths.
- Input and output directories are supplied through command-line arguments.
- Uses portable relative default paths when arguments are omitted.
- Supports both .nii and .nii.gz files.
- Produces inclusion and missing-file reports for quality control.

Expected directory structure
----------------------------
project/
├── classify_pattern_change_by_stage.py
├── data/
│   ├── onsets/
│   │   ├── Sub02/
│   │   │   └── csv_output/
│   │   │       └── cue_filtered_trials.csv
│   │   └── ...
│   └── pattern_change/
│       ├── Sub02_Pattern_change/
│       │   ├── pattern_change_trial_001.nii
│       │   ├── pattern_change_trial_002.nii
│       │   └── ...
│       └── ...
└── outputs/
    └── pattern_change_labeled/

Dependencies
------------
pip install pandas

Example
-------
python classify_pattern_change_by_stage.py \
    --onsets-root /path/to/onsets \
    --pattern-root /path/to/pattern_change \
    --output-root /path/to/pattern_change_labeled
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd


ALLOWED_LABELS = {
    "LN",
    "LE",
    "RN",
    "RE",
}

CSV_RELATIVE_PATH = (
    Path("csv_output")
    / "cue_filtered_trials.csv"
)

PATTERN_FILE_PREFIX = "pattern_change_trial_"

SUBJECT_PATTERN = re.compile(
    r"^Sub(\d+)$",
    flags=re.IGNORECASE,
)


def parse_arguments() -> argparse.Namespace:
    """Parse portable input, output, and copy settings."""
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Copy trial-wise pattern-change images into "
            "LN, LE, RN, and RE directories."
        )
    )

    parser.add_argument(
        "--onsets-root",
        type=Path,
        default=script_dir / "data" / "onsets",
        help=(
            "Root directory containing SubXX/csv_output/"
            "cue_filtered_trials.csv. "
            "Default: <script_dir>/data/onsets"
        ),
    )

    parser.add_argument(
        "--pattern-root",
        type=Path,
        default=script_dir / "data" / "pattern_change",
        help=(
            "Root directory containing SubXX_Pattern_change folders. "
            "Default: <script_dir>/data/pattern_change"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=script_dir / "outputs" / "pattern_change_labeled",
        help=(
            "Output directory for stage-labeled pattern-change images. "
            "Default: <script_dir>/outputs/pattern_change_labeled"
        ),
    )

    parser.add_argument(
        "--subject-ids",
        nargs="+",
        default=None,
        help=(
            "Optional participant identifiers, for example 02 03 04. "
            "When omitted, all Sub* folders under onsets_root are used."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing destination files.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Display planned operations without copying any files."
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
            f"Could not parse participant ID from: {value}"
        )

    return match.group(1)


def detect_subject_directories(
    onsets_root: Path,
) -> list[Path]:
    """Detect Sub* participant directories under the onset root."""
    subject_directories: list[Path] = []

    for path in onsets_root.iterdir():
        if not path.is_dir():
            continue

        if SUBJECT_PATTERN.fullmatch(path.name):
            subject_directories.append(path)

    return sorted(
        subject_directories,
        key=lambda path: natural_sort_key(path.name),
    )


def find_subject_directory(
    root: Path,
    subject_id: str,
) -> Path | None:
    """
    Locate a participant directory using common zero-padding variants.

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
            root
            / candidate_name
        )

        if candidate_path.is_dir():
            return candidate_path

    return None


def get_subject_directories(
    onsets_root: Path,
    requested_subject_ids: list[str] | None,
) -> list[Path]:
    """Return requested participant directories or detect all participants."""
    if requested_subject_ids is None:
        return detect_subject_directories(
            onsets_root
        )

    subject_directories: list[Path] = []

    for requested_id in requested_subject_ids:
        subject_id = normalize_subject_id(
            requested_id
        )

        subject_directory = find_subject_directory(
            onsets_root,
            subject_id,
        )

        if subject_directory is None:
            print(
                f"[Missing] Onset directory not found for "
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


def find_pattern_directory(
    pattern_root: Path,
    subject_name: str,
) -> Path | None:
    """
    Locate a participant pattern-change directory.

    Supported examples
    ------------------
    Sub02_Pattern_change
    Sub02_pattern_change
    """
    subject_id = normalize_subject_id(
        subject_name
    )

    participant_number = int(subject_id)

    subject_name_candidates = [
        subject_name,
        f"Sub{subject_id}",
        f"Sub{participant_number}",
        f"Sub{participant_number:02d}",
        f"Sub{participant_number:03d}",
    ]

    suffix_candidates = [
        "_Pattern_change",
        "_pattern_change",
    ]

    checked_paths: set[Path] = set()

    for candidate_subject_name in subject_name_candidates:
        for suffix in suffix_candidates:
            candidate_path = (
                pattern_root
                / f"{candidate_subject_name}{suffix}"
            )

            if candidate_path in checked_paths:
                continue

            checked_paths.add(candidate_path)

            if candidate_path.is_dir():
                return candidate_path

    return None


def safe_read_csv(
    csv_path: Path,
) -> pd.DataFrame | None:
    """
    Read cue_filtered_trials.csv and return standardized trial/label columns.
    """
    if not csv_path.is_file():
        print(
            f"[Missing] CSV file does not exist: {csv_path}"
        )
        return None

    dataframe = None
    read_errors: list[str] = []

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
            read_errors.append(
                f"{encoding}: {error}"
            )

        except Exception as error:
            print(
                f"[Skipped] Failed to read CSV file "
                f"{csv_path}: {error}"
            )
            return None

    if dataframe is None:
        print(
            f"[Skipped] Could not decode CSV file: {csv_path}"
        )

        for error_message in read_errors:
            print(
                f"  {error_message}"
            )

        return None

    column_lookup = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    if (
        "trial" not in column_lookup
        or "label" not in column_lookup
    ):
        print(
            f"[Skipped] CSV does not contain trial and label columns: "
            f"{csv_path}"
        )
        return None

    dataframe = dataframe.rename(
        columns={
            column_lookup["trial"]: "trial",
            column_lookup["label"]: "label",
        }
    )

    return dataframe[
        [
            "trial",
            "label",
        ]
    ].copy()


def normalize_trials_and_filter(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize labels and trial numbers.

    Operations
    ----------
    - Convert labels to uppercase.
    - Keep only LN, LE, RN, and RE rows.
    - Convert trial values to integers.
    - Remove invalid or non-positive trial indices.
    - Add a three-digit trial filename field.
    - Remove duplicated trial-label rows.
    """
    dataframe = dataframe.copy()

    dataframe["label"] = (
        dataframe["label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    dataframe = dataframe.loc[
        dataframe["label"].isin(
            ALLOWED_LABELS
        )
    ].copy()

    dataframe["trial"] = pd.to_numeric(
        dataframe["trial"],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=[
            "trial",
        ]
    ).copy()

    dataframe["trial"] = (
        dataframe["trial"]
        .astype(int)
    )

    dataframe = dataframe.loc[
        dataframe["trial"] >= 1
    ].copy()

    dataframe["trial_padded"] = (
        dataframe["trial"]
        .map(lambda value: f"{value:03d}")
    )

    dataframe = dataframe.drop_duplicates(
        subset=[
            "trial",
            "label",
        ],
        keep="first",
    )

    dataframe = dataframe.sort_values(
        by=[
            "trial",
            "label",
        ]
    ).reset_index(drop=True)

    return dataframe


def find_trial_file(
    pattern_directory: Path,
    trial_number: int,
) -> Path | None:
    """
    Locate a trial-wise pattern-change image.

    Search order
    ------------
    pattern_change_trial_001.nii
    pattern_change_trial_001.nii.gz
    pattern_change_trial_1.nii
    pattern_change_trial_1.nii.gz
    """
    padded_trial = f"{trial_number:03d}"

    candidate_names = [
        (
            f"{PATTERN_FILE_PREFIX}"
            f"{padded_trial}.nii"
        ),
        (
            f"{PATTERN_FILE_PREFIX}"
            f"{padded_trial}.nii.gz"
        ),
        (
            f"{PATTERN_FILE_PREFIX}"
            f"{trial_number}.nii"
        ),
        (
            f"{PATTERN_FILE_PREFIX}"
            f"{trial_number}.nii.gz"
        ),
    ]

    for candidate_name in candidate_names:
        candidate_path = (
            pattern_directory
            / candidate_name
        )

        if candidate_path.is_file():
            return candidate_path

    return None


def copy_one_trial(
    source_path: Path,
    output_directory: Path,
    *,
    overwrite: bool,
    dry_run: bool,
) -> tuple[bool, str]:
    """
    Copy one trial image.

    Returns
    -------
    success:
        Whether the destination is present or the copy succeeded.
    status:
        copied, overwritten, existing, or dry-run.
    """
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination_path = (
        output_directory
        / source_path.name
    )

    if destination_path.exists():
        if not overwrite:
            return True, "existing"

        status = "overwritten"

    else:
        status = "copied"

    if dry_run:
        print(
            f"[Dry run] {source_path} -> {destination_path}"
        )
        return True, "dry-run"

    shutil.copy2(
        source_path,
        destination_path,
    )

    return True, status


def clean_tsv_field(value: object) -> str:
    """Remove tabs and line breaks before writing a TSV field."""
    return (
        str(value)
        .replace("\t", " ")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def main() -> None:
    """Classify and copy pattern-change images by learning-stage label."""
    args = parse_arguments()

    onsets_root = (
        args.onsets_root
        .expanduser()
        .resolve()
    )

    pattern_root = (
        args.pattern_root
        .expanduser()
        .resolve()
    )

    output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    if not onsets_root.is_dir():
        raise NotADirectoryError(
            "Onset root directory does not exist: "
            f"{onsets_root}"
        )

    if not pattern_root.is_dir():
        raise NotADirectoryError(
            "Pattern-change root directory does not exist: "
            f"{pattern_root}"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    subject_directories = get_subject_directories(
        onsets_root,
        args.subject_ids,
    )

    if not subject_directories:
        raise RuntimeError(
            "No participant directories were found or selected."
        )

    print(
        "Onset root: "
        f"{onsets_root}"
    )
    print(
        "Pattern-change root: "
        f"{pattern_root}"
    )
    print(
        "Output root: "
        f"{output_root}"
    )
    print(
        "Participants scheduled: "
        f"{len(subject_directories)}"
    )
    print(
        "Overwrite existing files: "
        f"{args.overwrite}"
    )
    print(
        "Dry-run mode: "
        f"{args.dry_run}"
    )

    total_copied = 0
    total_existing = 0
    total_missing = 0

    subject_summary_rows: list[dict[str, object]] = []
    missing_file_rows: list[dict[str, object]] = []

    for subject_directory in subject_directories:
        subject_name = subject_directory.name

        csv_path = (
            subject_directory
            / CSV_RELATIVE_PATH
        )

        pattern_directory = find_pattern_directory(
            pattern_root,
            subject_name,
        )

        if pattern_directory is None:
            reason = "pattern-change directory not found"

            print(
                f"[Skipped] {subject_name}: {reason}."
            )

            subject_summary_rows.append(
                {
                    "participant_id": subject_name,
                    "n_labeled_rows": 0,
                    "n_copied": 0,
                    "n_existing": 0,
                    "n_missing": 0,
                    "status": reason,
                }
            )
            continue

        dataframe = safe_read_csv(
            csv_path
        )

        if dataframe is None:
            subject_summary_rows.append(
                {
                    "participant_id": subject_name,
                    "n_labeled_rows": 0,
                    "n_copied": 0,
                    "n_existing": 0,
                    "n_missing": 0,
                    "status": "CSV missing or unreadable",
                }
            )
            continue

        dataframe = normalize_trials_and_filter(
            dataframe
        )

        if dataframe.empty:
            reason = "no valid LN/LE/RN/RE rows"

            print(
                f"[Skipped] {subject_name}: {reason}."
            )

            subject_summary_rows.append(
                {
                    "participant_id": subject_name,
                    "n_labeled_rows": 0,
                    "n_copied": 0,
                    "n_existing": 0,
                    "n_missing": 0,
                    "status": reason,
                }
            )
            continue

        copied_for_subject = 0
        existing_for_subject = 0
        missing_for_subject = 0

        for row in dataframe.itertuples(
            index=False
        ):
            trial_number = int(
                row.trial
            )

            label = str(
                row.label
            )

            source_path = find_trial_file(
                pattern_directory,
                trial_number,
            )

            if source_path is None:
                expected_filename = (
                    f"{PATTERN_FILE_PREFIX}"
                    f"{trial_number:03d}.nii"
                )

                print(
                    f"[Missing] {subject_name}, {label}, "
                    f"trial {trial_number}: "
                    f"{pattern_directory / expected_filename}"
                )

                missing_for_subject += 1
                total_missing += 1

                missing_file_rows.append(
                    {
                        "participant_id": subject_name,
                        "label": label,
                        "trial": trial_number,
                        "expected_file": str(
                            pattern_directory
                            / expected_filename
                        ),
                    }
                )
                continue

            output_directory = (
                output_root
                / subject_name
                / label
            )

            success, status = copy_one_trial(
                source_path,
                output_directory,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )

            if not success:
                missing_for_subject += 1
                total_missing += 1
                continue

            if status == "existing":
                existing_for_subject += 1
                total_existing += 1
            else:
                copied_for_subject += 1
                total_copied += 1

        subject_summary_rows.append(
            {
                "participant_id": subject_name,
                "n_labeled_rows": len(dataframe),
                "n_copied": copied_for_subject,
                "n_existing": existing_for_subject,
                "n_missing": missing_for_subject,
                "status": "completed",
            }
        )

        print(
            f"[Completed] {subject_name}: "
            f"copied={copied_for_subject}, "
            f"existing={existing_for_subject}, "
            f"missing={missing_for_subject}."
        )

    summary_path = (
        output_root
        / "copy_summary.tsv"
    )

    summary_dataframe = pd.DataFrame(
        subject_summary_rows
    )

    summary_dataframe.to_csv(
        summary_path,
        sep="\t",
        index=False,
        encoding="utf-8",
    )

    missing_path = (
        output_root
        / "missing_pattern_files.tsv"
    )

    missing_dataframe = pd.DataFrame(
        missing_file_rows,
        columns=[
            "participant_id",
            "label",
            "trial",
            "expected_file",
        ],
    )

    missing_dataframe.to_csv(
        missing_path,
        sep="\t",
        index=False,
        encoding="utf-8",
    )

    print("\n========== Summary ==========")
    print(
        f"Files copied or overwritten: {total_copied}"
    )
    print(
        f"Existing files skipped:      {total_existing}"
    )
    print(
        f"Missing source files:        {total_missing}"
    )
    print(
        f"Participant summary:         {summary_path}"
    )
    print(
        f"Missing-file report:         {missing_path}"
    )
    print(
        f"Output directory:            {output_root}"
    )


if __name__ == "__main__":
    main()