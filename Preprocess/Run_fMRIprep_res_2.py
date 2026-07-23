#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batch-run fMRIPrep 24.1.1 using Docker.

Default processing settings
---------------------------
- Process each BIDS participant separately.
- Disable slice-timing correction globally.
- Optionally run ICA-AROMA using non-aggressive denoising.
- Request MNI152NLin2009cAsym:res-2 and T1w outputs.
- Preserve detailed console and file logs.
- Continue processing later participants when one participant fails.

Public-release version
----------------------
- Contains no user-specific paths, drive letters, or machine names.
- Host paths are supplied through command-line arguments.
- Docker container paths use fixed generic locations:
    /data
    /out
    /work
    /opt/freesurfer/license.txt

Default directory structure
---------------------------
project/
├── run_fmriprep_docker.py
├── data/
│   └── bids/
├── config/
│   └── license.txt
├── outputs/
│   └── fmriprep/
└── work/
    └── fmriprep/

Example
-------
python run_fmriprep_docker.py \
    --bids-dir /path/to/bids \
    --output-dir /path/to/fmriprep_output \
    --work-dir /path/to/fmriprep_work \
    --fs-license /path/to/license.txt
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Sequence


DEFAULT_IMAGE = "nipreps/fmriprep:24.1.1"
DEFAULT_OUTPUT_SPACES = [
    "MNI152NLin2009cAsym:res-2",
    "T1w",
]


def parse_arguments() -> argparse.Namespace:
    """Parse portable path and fMRIPrep settings."""
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Run fMRIPrep participant-by-participant using Docker."
        )
    )

    parser.add_argument(
        "--bids-dir",
        type=Path,
        default=script_dir / "data" / "bids",
        help=(
            "Input BIDS dataset directory. "
            "Default: <script_dir>/data/bids"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "outputs" / "fmriprep",
        help=(
            "fMRIPrep output directory. "
            "Default: <script_dir>/outputs/fmriprep"
        ),
    )

    parser.add_argument(
        "--work-dir",
        type=Path,
        default=script_dir / "work" / "fmriprep",
        help=(
            "fMRIPrep working directory. "
            "Default: <script_dir>/work/fmriprep"
        ),
    )

    parser.add_argument(
        "--fs-license",
        type=Path,
        default=script_dir / "config" / "license.txt",
        help=(
            "FreeSurfer license file. "
            "Default: <script_dir>/config/license.txt"
        ),
    )

    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help=(
            "Docker image used for fMRIPrep. "
            f"Default: {DEFAULT_IMAGE}"
        ),
    )

    parser.add_argument(
        "--docker-command",
        default="docker",
        help=(
            "Docker executable or command. Default: docker"
        ),
    )

    parser.add_argument(
        "--nthreads",
        type=int,
        default=15,
        help="Number of processing threads. Default: 15",
    )

    parser.add_argument(
        "--omp-nthreads",
        type=int,
        default=None,
        help=(
            "Maximum OpenMP threads per process. "
            "When omitted, fMRIPrep determines the value."
        ),
    )

    parser.add_argument(
        "--mem-mb",
        type=int,
        default=32000,
        help="Memory limit in MB. Default: 32000",
    )

    parser.add_argument(
        "--output-spaces",
        nargs="+",
        default=DEFAULT_OUTPUT_SPACES,
        help=(
            "Output spaces passed to fMRIPrep. "
            "Default: MNI152NLin2009cAsym:res-2 T1w"
        ),
    )

    parser.add_argument(
        "--participant-labels",
        nargs="+",
        default=None,
        help=(
            "Optional participant labels to process, without the "
            "sub- prefix. When omitted, all BIDS participants are used."
        ),
    )

    aroma_group = parser.add_mutually_exclusive_group()

    aroma_group.add_argument(
        "--use-aroma",
        dest="use_aroma",
        action="store_true",
        help="Run ICA-AROMA using non-aggressive denoising.",
    )

    aroma_group.add_argument(
        "--no-aroma",
        dest="use_aroma",
        action="store_false",
        help="Do not run ICA-AROMA.",
    )

    parser.set_defaults(use_aroma=True)

    slice_timing_group = parser.add_mutually_exclusive_group()

    slice_timing_group.add_argument(
        "--ignore-slicetiming",
        dest="ignore_slicetiming",
        action="store_true",
        help="Disable slice-timing correction.",
    )

    slice_timing_group.add_argument(
        "--use-slicetiming",
        dest="ignore_slicetiming",
        action="store_false",
        help="Allow fMRIPrep to perform slice-timing correction.",
    )

    parser.set_defaults(ignore_slicetiming=True)

    validation_group = parser.add_mutually_exclusive_group()

    validation_group.add_argument(
        "--skip-bids-validation",
        dest="skip_bids_validation",
        action="store_true",
        help="Skip BIDS validation inside fMRIPrep.",
    )

    validation_group.add_argument(
        "--run-bids-validation",
        dest="skip_bids_validation",
        action="store_false",
        help="Allow fMRIPrep to perform BIDS validation.",
    )

    parser.set_defaults(skip_bids_validation=True)

    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help=(
            "Optional log-file path. When omitted, a timestamped log "
            "is created under <output-dir>/logs."
        ),
    )

    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help=(
            "Retain each participant's working directory after successful "
            "completion. This script never removes work files unless this "
            "option is omitted."
        ),
    )

    return parser.parse_args()


def configure_logging(log_file: Path) -> None:
    """Configure simultaneous console and file logging."""
    log_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def natural_sort_key(value: str) -> list[object]:
    """Sort labels naturally, for example 2 before 10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def ensure_dataset_description(bids_root: Path) -> None:
    """Create a minimal dataset_description.json when it is missing."""
    description_file = (
        bids_root
        / "dataset_description.json"
    )

    if description_file.is_file():
        return

    logging.warning(
        "dataset_description.json is missing; creating a minimal file."
    )

    metadata = {
        "Name": "BIDS Dataset",
        "BIDSVersion": "1.8.0",
    }

    with description_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )


def normalize_participant_label(label: str) -> str:
    """Remove an optional sub- prefix from a participant label."""
    normalized = re.sub(
        r"^sub-",
        "",
        str(label).strip(),
        flags=re.IGNORECASE,
    )

    if not re.fullmatch(r"[A-Za-z0-9]+", normalized):
        raise ValueError(
            f"Invalid participant label: {label!r}"
        )

    return normalized


def find_bids_participants(bids_root: Path) -> list[str]:
    """Find participant labels from sub-* directories."""
    participant_labels = [
        path.name.removeprefix("sub-")
        for path in bids_root.iterdir()
        if (
            path.is_dir()
            and re.fullmatch(
                r"sub-[A-Za-z0-9]+",
                path.name,
            )
        )
    ]

    return sorted(
        participant_labels,
        key=natural_sort_key,
    )


def select_participants(
    available_labels: Sequence[str],
    requested_labels: Sequence[str] | None,
) -> list[str]:
    """Select all participants or a requested participant subset."""
    if requested_labels is None:
        return list(available_labels)

    normalized_requested = [
        normalize_participant_label(label)
        for label in requested_labels
    ]

    available_set = set(available_labels)

    missing_labels = [
        label
        for label in normalized_requested
        if label not in available_set
    ]

    if missing_labels:
        raise ValueError(
            "The following requested participant labels were not found "
            f"in the BIDS dataset: {', '.join(missing_labels)}"
        )

    return sorted(
        set(normalized_requested),
        key=natural_sort_key,
    )


def docker_volume_argument(
    host_path: Path,
    container_path: str,
    read_only: bool = False,
) -> str:
    """Construct one Docker bind-volume argument."""
    suffix = ":ro" if read_only else ""

    return (
        f"{host_path}"
        f":{container_path}"
        f"{suffix}"
    )


def format_command(command: Sequence[str]) -> str:
    """Format a command for readable logging."""
    return subprocess.list2cmdline(
        list(command)
    )


def build_fmriprep_command(
    *,
    docker_command: str,
    image: str,
    bids_dir: Path,
    output_dir: Path,
    work_dir: Path,
    fs_license: Path,
    participant_label: str,
    nthreads: int,
    omp_nthreads: int | None,
    mem_mb: int,
    output_spaces: Sequence[str],
    use_aroma: bool,
    ignore_slicetiming: bool,
    skip_bids_validation: bool,
) -> list[str]:
    """Build the Docker command for one participant."""
    container_bids_dir = "/data"
    container_output_dir = "/out"
    container_work_dir = "/work"
    container_license_file = (
        "/opt/freesurfer/license.txt"
    )

    command = [
        docker_command,
        "run",
        "--rm",

        "-v",
        docker_volume_argument(
            bids_dir,
            container_bids_dir,
            read_only=True,
        ),

        "-v",
        docker_volume_argument(
            output_dir,
            container_output_dir,
        ),

        "-v",
        docker_volume_argument(
            work_dir,
            container_work_dir,
        ),

        "-v",
        docker_volume_argument(
            fs_license,
            container_license_file,
            read_only=True,
        ),

        image,

        container_bids_dir,
        container_output_dir,
        "participant",

        "--participant-label",
        participant_label,

        "--fs-license-file",
        container_license_file,

        "--nthreads",
        str(nthreads),

        "--mem_mb",
        str(mem_mb),

        "--work-dir",
        container_work_dir,

        "--output-spaces",
        *output_spaces,
    ]

    if omp_nthreads is not None:
        command.extend(
            [
                "--omp-nthreads",
                str(omp_nthreads),
            ]
        )

    if skip_bids_validation:
        command.append(
            "--skip-bids-validation"
        )

    if use_aroma:
        command.extend(
            [
                "--use-aroma",
                "nonaggr",
            ]
        )

    if ignore_slicetiming:
        command.extend(
            [
                "--ignore",
                "slicetiming",
            ]
        )

    return command


def save_failure_summary(
    output_dir: Path,
    failures: list[tuple[str, str]],
) -> Path:
    """Write a participant-level failure summary."""
    failure_file = (
        output_dir
        / "fmriprep_failed_participants.tsv"
    )

    with failure_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        file.write(
            "participant_id\terror\n"
        )

        for participant_label, message in failures:
            clean_message = (
                str(message)
                .replace("\t", " ")
                .replace("\r", " ")
                .replace("\n", " ")
            )

            file.write(
                f"sub-{participant_label}\t"
                f"{clean_message}\n"
            )

    return failure_file


def main() -> None:
    """Run fMRIPrep sequentially for all selected participants."""
    args = parse_arguments()

    bids_dir = (
        args.bids_dir
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    work_dir = (
        args.work_dir
        .expanduser()
        .resolve()
    )

    fs_license = (
        args.fs_license
        .expanduser()
        .resolve()
    )

    if not bids_dir.is_dir():
        raise NotADirectoryError(
            "BIDS input directory does not exist: "
            f"{bids_dir}"
        )

    if not fs_license.is_file():
        raise FileNotFoundError(
            "FreeSurfer license file does not exist: "
            f"{fs_license}"
        )

    if args.nthreads < 1:
        raise ValueError(
            "--nthreads must be at least 1."
        )

    if (
        args.omp_nthreads is not None
        and args.omp_nthreads < 1
    ):
        raise ValueError(
            "--omp-nthreads must be at least 1."
        )

    if args.mem_mb < 1:
        raise ValueError(
            "--mem-mb must be greater than zero."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.log_file is None:
        timestamp = datetime.datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        log_file = (
            output_dir
            / "logs"
            / f"fmriprep_batch_{timestamp}.log"
        )
    else:
        log_file = (
            args.log_file
            .expanduser()
            .resolve()
        )

    configure_logging(log_file)

    logging.info(
        "BIDS input directory: %s",
        bids_dir,
    )
    logging.info(
        "fMRIPrep output directory: %s",
        output_dir,
    )
    logging.info(
        "Working directory: %s",
        work_dir,
    )
    logging.info(
        "FreeSurfer license: %s",
        fs_license,
    )
    logging.info(
        "Docker image: %s",
        args.image,
    )
    logging.info(
        "Output spaces: %s",
        ", ".join(args.output_spaces),
    )
    logging.info(
        "ICA-AROMA enabled: %s",
        args.use_aroma,
    )
    logging.info(
        "Slice-timing correction ignored: %s",
        args.ignore_slicetiming,
    )
    logging.info(
        "BIDS validation skipped: %s",
        args.skip_bids_validation,
    )

    ensure_dataset_description(
        bids_dir
    )

    available_participants = (
        find_bids_participants(
            bids_dir
        )
    )

    if not available_participants:
        raise RuntimeError(
            "No participant directories matching sub-* "
            f"were found in: {bids_dir}"
        )

    participant_labels = select_participants(
        available_participants,
        args.participant_labels,
    )

    logging.info(
        "Detected %d participants: %s",
        len(participant_labels),
        ", ".join(
            f"sub-{label}"
            for label in participant_labels
        ),
    )

    successful_participants: list[str] = []
    failed_participants: list[tuple[str, str]] = []

    for participant_label in participant_labels:
        logging.info(
            "============================================================"
        )
        logging.info(
            "Starting fMRIPrep for sub-%s",
            participant_label,
        )

        command = build_fmriprep_command(
            docker_command=args.docker_command,
            image=args.image,
            bids_dir=bids_dir,
            output_dir=output_dir,
            work_dir=work_dir,
            fs_license=fs_license,
            participant_label=participant_label,
            nthreads=args.nthreads,
            omp_nthreads=args.omp_nthreads,
            mem_mb=args.mem_mb,
            output_spaces=args.output_spaces,
            use_aroma=args.use_aroma,
            ignore_slicetiming=args.ignore_slicetiming,
            skip_bids_validation=args.skip_bids_validation,
        )

        logging.info(
            "Running command:\n%s",
            format_command(command),
        )

        try:
            subprocess.run(
                command,
                check=True,
            )

            successful_participants.append(
                participant_label
            )

            logging.info(
                "Completed sub-%s successfully.",
                participant_label,
            )

        except FileNotFoundError as error:
            message = (
                f"Docker executable was not found: {error}"
            )

            failed_participants.append(
                (
                    participant_label,
                    message,
                )
            )

            logging.error(
                "sub-%s failed: %s",
                participant_label,
                message,
            )

            # Docker is unavailable, so later participants would also fail.
            break

        except subprocess.CalledProcessError as error:
            message = (
                f"fMRIPrep returned exit code "
                f"{error.returncode}"
            )

            failed_participants.append(
                (
                    participant_label,
                    message,
                )
            )

            logging.error(
                "sub-%s failed: %s",
                participant_label,
                message,
            )

        except Exception as error:
            message = (
                f"Unexpected error: {error}"
            )

            failed_participants.append(
                (
                    participant_label,
                    message,
                )
            )

            logging.exception(
                "Unexpected error while processing sub-%s",
                participant_label,
            )

    failure_file = save_failure_summary(
        output_dir,
        failed_participants,
    )

    logging.info(
        "============================================================"
    )
    logging.info(
        "Batch processing finished."
    )
    logging.info(
        "Successful participants: %d",
        len(successful_participants),
    )
    logging.info(
        "Failed participants: %d",
        len(failed_participants),
    )
    logging.info(
        "Failure summary: %s",
        failure_file,
    )
    logging.info(
        "Complete log: %s",
        log_file,
    )


if __name__ == "__main__":
    main()