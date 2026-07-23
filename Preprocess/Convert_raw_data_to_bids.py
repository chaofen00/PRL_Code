#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert an RL Go/NoGo MRI dataset from DICOM folders to BIDS format.

Public-release version:
- No user names, drive letters, or machine-specific absolute paths are stored.
- Input, output, and dcm2niix locations are provided through command-line arguments.
- The original conversion workflow and detailed logging are retained.

Expected raw-data structure
---------------------------
raw_data/
├── Sub001/
│   ├── RUN1/
│   ├── RUN2/
│   ├── RUN3/
│   └── T1/
├── Sub002/
│   └── ...
└── ...

Example
-------
python convert_rl_gonogo_to_bids.py \
    --dcm2niix /path/to/dcm2niix \
    --raw-data-dir /path/to/raw_data \
    --bids-output-dir /path/to/bids_output
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import re
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Any


def parse_arguments() -> argparse.Namespace:
    """Parse portable input, output, and metadata settings."""
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Convert RL Go/NoGo DICOM folders to a BIDS dataset."
    )

    parser.add_argument(
        "--dcm2niix",
        default="dcm2niix",
        help=(
            "Path to the dcm2niix executable, or 'dcm2niix' when it is "
            "available on the system PATH."
        ),
    )

    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=script_dir / "data" / "raw",
        help=(
            "Root directory containing SubXXX/T1 and "
            "SubXXX/RUN1-RUN3 folders."
        ),
    )

    parser.add_argument(
        "--bids-output-dir",
        type=Path,
        default=script_dir / "outputs" / "bids",
        help="Destination directory for the BIDS dataset.",
    )

    parser.add_argument(
        "--tr",
        type=float,
        default=2.2,
        help=(
            "Repetition time in seconds written to functional "
            "JSON sidecars."
        ),
    )

    parser.add_argument(
        "--task-name",
        default="gonogo",
        help=(
            "BIDS task name used in functional filenames "
            "and JSON metadata."
        ),
    )

    parser.add_argument(
        "--dataset-name",
        default="RL GoNoGo BIDS Dataset",
        help="Dataset name written to dataset_description.json.",
    )

    parser.add_argument(
        "--bids-version",
        default="1.8.0",
        help="BIDS version written to dataset_description.json.",
    )

    parser.add_argument(
        "--author",
        action="append",
        default=[],
        help=(
            "Dataset author written to dataset_description.json. "
            "Repeat this option to provide multiple authors."
        ),
    )

    return parser.parse_args()


def configure_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )


def natural_sort_key(value: str) -> list[Any]:
    """Sort participant folders naturally, for example Sub2 before Sub10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def init_anomaly_log(anomaly_log_file: Path) -> None:
    """Back up an existing anomaly log before starting a new conversion."""
    anomaly_log_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if anomaly_log_file.exists():
        timestamp = datetime.datetime.now().strftime(
            "%Y%m%d%H%M%S"
        )

        backup_path = anomaly_log_file.with_name(
            f"{anomaly_log_file.name}.{timestamp}.bak"
        )

        shutil.move(
            str(anomaly_log_file),
            str(backup_path),
        )


def log_anomaly(
    subject_id: str,
    message: str,
    anomaly_log_file: Path,
) -> None:
    """Append one participant-level anomaly to the anomaly log."""
    timestamp = datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    line = (
        f"[{timestamp}] "
        f"sub-{subject_id}: "
        f"{message}\n"
    )

    with anomaly_log_file.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(line)

    logging.warning(
        "sub-%s: %s",
        subject_id,
        message,
    )


def check_folder_has_dicom(folder: Path) -> bool:
    """
    Return True when a folder contains at least one likely DICOM file.

    DICOM exports may use .dcm, .ima, or no filename extension. The fallback
    to any non-empty file preserves compatibility with scanner exports that
    do not use standard extensions.
    """
    if not folder.is_dir():
        return False

    for path in folder.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() in {
            ".dcm",
            ".ima",
        }:
            return True

        try:
            if path.stat().st_size > 0:
                return True
        except OSError:
            continue

    return False


def check_subject_integrity(
    raw_data_dir: Path,
    subject_dir_name: str,
) -> dict[str, Any]:
    """Check whether T1 and all three task-run folders contain source files."""
    source_dir = (
        raw_data_dir
        / subject_dir_name
    )

    has_t1 = check_folder_has_dicom(
        source_dir / "T1"
    )

    has_runs = {
        run: check_folder_has_dicom(
            source_dir / run
        )
        for run in (
            "RUN1",
            "RUN2",
            "RUN3",
        )
    }

    issues: list[str] = []

    if not has_t1:
        issues.append("missing T1")

    for run, is_present in has_runs.items():
        if not is_present:
            issues.append(
                f"missing {run}"
            )

    return {
        "T1": has_t1,
        **has_runs,
        "error": (
            "; ".join(issues)
            if issues
            else None
        ),
    }


def create_bids_structure(
    bids_output_dir: Path,
    subject_ids: list[str],
    dataset_name: str,
    bids_version: str,
    authors: list[str],
) -> None:
    """Create top-level BIDS metadata files."""
    logging.info(
        "Creating top-level BIDS directories and metadata files"
    )

    bids_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_description: dict[str, Any] = {
        "Name": dataset_name,
        "BIDSVersion": bids_version,
    }

    if authors:
        dataset_description["Authors"] = authors

    try:
        with (
            bids_output_dir
            / "dataset_description.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                dataset_description,
                file,
                indent=4,
                ensure_ascii=False,
            )

    except Exception as error:
        logging.error(
            "Failed to write dataset_description.json: %s",
            error,
        )
        logging.error(
            traceback.format_exc()
        )

    try:
        with (
            bids_output_dir
            / "participants.tsv"
        ).open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            file.write(
                "participant_id\tsex\tage\n"
            )

            for subject_id in subject_ids:
                file.write(
                    f"sub-{subject_id}\tn/a\tn/a\n"
                )

    except Exception as error:
        logging.error(
            "Failed to write participants.tsv: %s",
            error,
        )
        logging.error(
            traceback.format_exc()
        )

    try:
        with (
            bids_output_dir
            / "README"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            file.write(
                "# RL Go/NoGo BIDS Dataset\n\n"
                "This dataset contains functional MRI data "
                "from a Go/NoGo task and corresponding "
                "T1-weighted anatomical images.\n"
            )

    except Exception as error:
        logging.error(
            "Failed to write README: %s",
            error,
        )
        logging.error(
            traceback.format_exc()
        )


def patch_json(
    json_file: Path,
    repetition_time: float,
    task_name: str | None = None,
) -> None:
    """Patch selected fields in a dcm2niix-generated JSON sidecar."""
    try:
        with json_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        changed = False

        if (
            metadata.get("RepetitionTime")
            != repetition_time
        ):
            logging.info(
                "Updating %s: RepetitionTime %r -> %s",
                json_file,
                metadata.get("RepetitionTime"),
                repetition_time,
            )

            metadata["RepetitionTime"] = (
                repetition_time
            )
            changed = True

        if "AcquisitionDuration" in metadata:
            logging.info(
                "Removing AcquisitionDuration from %s",
                json_file,
            )

            del metadata["AcquisitionDuration"]
            changed = True

        if (
            task_name
            and metadata.get("TaskName")
            != task_name
        ):
            metadata["TaskName"] = task_name
            changed = True

        if changed:
            with json_file.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    metadata,
                    file,
                    indent=4,
                    ensure_ascii=False,
                )

            logging.info(
                "Updated JSON sidecar: %s",
                json_file.name,
            )

    except Exception as error:
        logging.error(
            "JSON patch failed for %s: %s",
            json_file,
            error,
        )
        logging.error(
            traceback.format_exc()
        )


def cleanup_extra_t1w(
    bids_anat_dir: Path,
    prefix: str,
) -> bool:
    """Remove duplicate T1 outputs with dcm2niix suffixes a-d."""
    base = (
        bids_anat_dir
        / f"{prefix}_T1w"
    )

    for extension in (
        ".nii.gz",
        ".json",
    ):
        for suffix in (
            "a",
            "b",
            "c",
            "d",
        ):
            extra_file = Path(
                f"{base}{suffix}{extension}"
            )

            if extra_file.exists():
                logging.info(
                    "Removing duplicate T1 output: %s",
                    extra_file,
                )
                extra_file.unlink()

    return (
        Path(f"{base}.nii.gz").exists()
        and Path(f"{base}.json").exists()
    )


def validate_subject_conversion(
    subject_id: str,
    bids_subject_dir: Path,
    task_name: str,
    anomaly_log_file: Path,
) -> None:
    """Check that the expected anatomical and functional files exist."""
    required_files = [
        (
            bids_subject_dir
            / "anat"
            / f"sub-{subject_id}_T1w.nii.gz"
        ),
        (
            bids_subject_dir
            / "anat"
            / f"sub-{subject_id}_T1w.json"
        ),
    ]

    for run_index in (
        1,
        2,
        3,
    ):
        run_stem = (
            f"sub-{subject_id}_"
            f"task-{task_name}_"
            f"run-{run_index:02d}_bold"
        )

        required_files.extend(
            [
                (
                    bids_subject_dir
                    / "func"
                    / f"{run_stem}.nii.gz"
                ),
                (
                    bids_subject_dir
                    / "func"
                    / f"{run_stem}.json"
                ),
            ]
        )

    for path in required_files:
        if not path.exists():
            relative_path = path.relative_to(
                bids_subject_dir
            )

            log_anomaly(
                subject_id,
                f"missing file: {relative_path}",
                anomaly_log_file,
            )


def run_dcm2niix(
    dcm2niix_command: str,
    source_dir: Path,
    output_dir: Path,
    output_name: str,
) -> subprocess.CompletedProcess[str]:
    """Run dcm2niix for one source directory."""
    command = [
        dcm2niix_command,
        "-z",
        "y",
        "-f",
        output_name,
        "-o",
        str(output_dir),
        str(source_dir),
    ]

    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )


def convert_subject(
    subject_dir_name: str,
    subject_id: str,
    raw_data_dir: Path,
    bids_output_dir: Path,
    dcm2niix_command: str,
    repetition_time: float,
    task_name: str,
    anomaly_log_file: Path,
) -> None:
    """Convert one participant from DICOM folders to BIDS files."""
    source_dir = (
        raw_data_dir
        / subject_dir_name
    )

    bids_subject_dir = (
        bids_output_dir
        / f"sub-{subject_id}"
    )

    anat_dir = (
        bids_subject_dir
        / "anat"
    )

    func_dir = (
        bids_subject_dir
        / "func"
    )

    anat_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    func_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # T1-weighted anatomical image
    # ========================================================

    t1_source = (
        source_dir
        / "T1"
    )

    t1_output_name = (
        f"sub-{subject_id}_T1w"
    )

    try:
        logging.info(
            "Running dcm2niix for T1: %s -> %s",
            t1_source,
            anat_dir,
        )

        result = run_dcm2niix(
            dcm2niix_command,
            t1_source,
            anat_dir,
            t1_output_name,
        )

        logging.info(
            "dcm2niix T1 completed.\n"
            "stdout:\n%s\n"
            "stderr:\n%s",
            result.stdout,
            result.stderr,
        )

    except subprocess.CalledProcessError as error:
        logging.error(
            "dcm2niix T1 failed: %s\n"
            "stdout:\n%s\n"
            "stderr:\n%s",
            error,
            error.stdout,
            error.stderr,
        )

        log_anomaly(
            subject_id,
            f"dcm2niix T1 failed: {error}",
            anomaly_log_file,
        )
        return

    except FileNotFoundError as error:
        logging.error(
            "dcm2niix executable was not found: %s",
            error,
        )

        log_anomaly(
            subject_id,
            (
                "dcm2niix executable was not found: "
                f"{error}"
            ),
            anomaly_log_file,
        )
        return

    cleanup_extra_t1w(
        anat_dir,
        f"sub-{subject_id}",
    )

    # ========================================================
    # Functional task runs
    # ========================================================

    for run_index, run_folder in enumerate(
        (
            "RUN1",
            "RUN2",
            "RUN3",
        ),
        start=1,
    ):
        run_source = (
            source_dir
            / run_folder
        )

        run_output_name = (
            f"sub-{subject_id}_"
            f"task-{task_name}_"
            f"run-{run_index:02d}_bold"
        )

        try:
            logging.info(
                "Running dcm2niix for task run %d: %s -> %s",
                run_index,
                run_source,
                func_dir,
            )

            result = run_dcm2niix(
                dcm2niix_command,
                run_source,
                func_dir,
                run_output_name,
            )

            logging.info(
                "dcm2niix task run %d completed.\n"
                "stdout:\n%s\n"
                "stderr:\n%s",
                run_index,
                result.stdout,
                result.stderr,
            )

        except subprocess.CalledProcessError as error:
            logging.error(
                "dcm2niix task run %d failed: %s\n"
                "stdout:\n%s\n"
                "stderr:\n%s",
                run_index,
                error,
                error.stdout,
                error.stderr,
            )

            log_anomaly(
                subject_id,
                (
                    f"dcm2niix task run "
                    f"{run_index} failed: {error}"
                ),
                anomaly_log_file,
            )
            continue

        except FileNotFoundError as error:
            logging.error(
                "dcm2niix executable was not found: %s",
                error,
            )

            log_anomaly(
                subject_id,
                (
                    "dcm2niix executable was not found: "
                    f"{error}"
                ),
                anomaly_log_file,
            )
            return

        run_nifti = (
            func_dir
            / f"{run_output_name}.nii.gz"
        )

        run_json = (
            func_dir
            / f"{run_output_name}.json"
        )

        try:
            if (
                run_nifti.exists()
                and run_json.exists()
            ):
                patch_json(
                    run_json,
                    repetition_time,
                    task_name=task_name,
                )

            else:
                log_anomaly(
                    subject_id,
                    (
                        f"task run {run_index} "
                        "NIfTI or JSON is missing"
                    ),
                    anomaly_log_file,
                )

        except Exception as error:
            logging.error(
                (
                    "Task run %d NIfTI/JSON "
                    "processing failed: %s"
                ),
                run_index,
                error,
            )

            logging.error(
                traceback.format_exc()
            )

            log_anomaly(
                subject_id,
                (
                    f"task run {run_index} "
                    f"NIfTI/JSON processing failed: {error}"
                ),
                anomaly_log_file,
            )

    # ========================================================
    # Final integrity check
    # ========================================================

    try:
        validate_subject_conversion(
            subject_id,
            bids_subject_dir,
            task_name,
            anomaly_log_file,
        )

    except Exception as error:
        logging.error(
            (
                "validate_subject_conversion "
                "failed for sub-%s: %s"
            ),
            subject_id,
            error,
        )

        logging.error(
            traceback.format_exc()
        )


def main() -> None:
    """Run the complete DICOM-to-BIDS conversion."""
    configure_logging()
    args = parse_arguments()

    raw_data_dir = (
        args.raw_data_dir
        .expanduser()
        .resolve()
    )

    bids_output_dir = (
        args.bids_output_dir
        .expanduser()
        .resolve()
    )

    anomaly_log_file = (
        bids_output_dir
        / "anomaly_log.txt"
    )

    if not raw_data_dir.is_dir():
        raise NotADirectoryError(
            "Raw-data directory does not exist: "
            f"{raw_data_dir}"
        )

    bids_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    init_anomaly_log(
        anomaly_log_file
    )

    try:
        subject_directories = [
            path.name
            for path in raw_data_dir.iterdir()
            if (
                path.is_dir()
                and re.fullmatch(
                    r"Sub\d+",
                    path.name,
                    flags=re.IGNORECASE,
                )
            )
        ]

    except Exception as error:
        logging.error(
            "Failed to read raw-data directory: %s",
            error,
        )

        logging.error(
            traceback.format_exc()
        )
        return

    subject_directories.sort(
        key=natural_sort_key
    )

    subject_ids = [
        re.sub(
            r"^Sub",
            "",
            directory_name,
            flags=re.IGNORECASE,
        ).zfill(3)
        for directory_name in subject_directories
    ]

    logging.info(
        "Found %d participant directories: %s",
        len(subject_directories),
        ", ".join(subject_directories),
    )

    valid_subject_ids: list[str] = []

    for directory_name, subject_id in zip(
        subject_directories,
        subject_ids,
    ):
        try:
            integrity = check_subject_integrity(
                raw_data_dir,
                directory_name,
            )

            if integrity["error"]:
                log_anomaly(
                    subject_id,
                    integrity["error"],
                    anomaly_log_file,
                )

            else:
                valid_subject_ids.append(
                    subject_id
                )

        except Exception as error:
            logging.error(
                (
                    "check_subject_integrity"
                    "(%s, %s) failed: %s"
                ),
                directory_name,
                subject_id,
                error,
            )

            logging.error(
                traceback.format_exc()
            )

    create_bids_structure(
        bids_output_dir,
        valid_subject_ids,
        args.dataset_name,
        args.bids_version,
        args.author,
    )

    for directory_name, subject_id in zip(
        subject_directories,
        subject_ids,
    ):
        if subject_id not in valid_subject_ids:
            continue

        logging.info(
            "Starting conversion for sub-%s",
            subject_id,
        )

        try:
            convert_subject(
                directory_name,
                subject_id,
                raw_data_dir,
                bids_output_dir,
                args.dcm2niix,
                args.tr,
                args.task_name,
                anomaly_log_file,
            )

        except Exception as error:
            logging.error(
                "convert_subject(%s, %s) failed: %s",
                directory_name,
                subject_id,
                error,
            )

            logging.error(
                traceback.format_exc()
            )

            log_anomaly(
                subject_id,
                f"conversion failed: {error}",
                anomaly_log_file,
            )

    logging.info(
        (
            "Conversion finished for all complete participants. "
            "See %s for anomaly details."
        ),
        anomaly_log_file,
    )


if __name__ == "__main__":
    main()