"""Legacy OCR script. DEPRECATED: use ``src.ocr`` module instead.

This script is kept for backward compatibility.  New code should use::

    from src.ocr import OCRRegistry, detect_devices, OCRStatusTracker

    model = OCRRegistry.get_model("marker", output_dir="data/ocr_results")
    result = model.process_single("paper.pdf")

See ``src/ocr/README.md`` for the extension guide.
"""

import csv
import json
import logging
import os
import shutil
import subprocess
import time
import warnings
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Dict, List, Tuple

import psutil
import torch

from config_manager import config

warnings.warn(
    "src/chunked-ocr.py is deprecated. Use the src.ocr module instead. "
    "See src/ocr/README.md for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)


def CollectInputFiles(input_dir: str) -> list[str]:
    """
    Collect all input files from the input directory.
    """
    return [
        f
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".pdf")
    ]


def DeviceDetection() -> Tuple[int, Dict[int, int]]:
    """
    Detect the device type.
    List GPU numbers and the memory of each GPU.
    Calculate optimal workers per GPU based on 3.5GB per worker.
    """
    gpu_count = torch.cuda.device_count()
    gpu_workers = {}

    print(f"GPU numbers: {gpu_count}")

    if gpu_count > 0:
        for i in range(gpu_count):
            gpu_memory_gb = (
                torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024
            )
            # Calculate workers based on 3.5GB per worker with some headroom
            workers_per_gpu = max(1, int(gpu_memory_gb / 3.5))
            gpu_workers[i] = workers_per_gpu
            print(f"GPU {i}: {gpu_memory_gb:.2f} GB, Workers: {workers_per_gpu}")
    else:
        print("No GPUs detected, using CPU mode")
        # For CPU, use number of cores with some limit
        cpu_cores = psutil.cpu_count(logical=False)
        gpu_workers[0] = min(cpu_cores, 4)  # Limit CPU workers

    print(f"CPU memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.2f} GB")

    return gpu_count, gpu_workers


# Global status tracking
status_tracker = {
    "task_name": "chunked_ocr",
    "total_files": 0,
    "completed_files": 0,
    "failed_files": 0,
    "start_time": None,
    "current_time": None,
    "time_cost": 0,
}

# Global results list for incremental saving
global_results = []
csv_save_path = None


def SaveResultsToCsv(results: List[Dict], csv_path: str, append: bool = False):
    """
    Save OCR results to CSV file. Can append to existing file.
    """
    mode = "a" if append and Path(csv_path).exists() else "w"
    write_header = not (append and Path(csv_path).exists())

    with open(csv_path, mode, newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "paper_name",
            "original_pdf_path",
            "ocr_status",
            "ocr_time",
            "ocr_result_path",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(results)


def ProcessPdfBatch(
    pdf_files: List[str], output_dir: str, gpu_count: int, workers_per_gpu: int
) -> List[Dict]:
    """
    Process PDFs using marker CLI in batch mode.
    """
    global global_results, csv_save_path
    results = []

    # Create temporary input directory for batch processing
    temp_input_dir = Path(output_dir).parent / "temp_pdf_batch"
    temp_input_dir.mkdir(exist_ok=True)

    # Copy PDFs to temp directory
    for pdf_path in pdf_files:
        shutil.copy2(pdf_path, temp_input_dir)

    # Prepare environment variables
    env = os.environ.copy()
    env["NUM_DEVICES"] = str(max(1, gpu_count))
    env["NUM_WORKERS"] = str(workers_per_gpu)

    # Run marker command
    cmd = ["marker_chunk_convert", str(temp_input_dir), str(output_dir)]

    try:
        logging.info(
            f"Running marker with {gpu_count} GPUs and {workers_per_gpu} workers per GPU"
        )
        logging.info(f"Command: {' '.join(cmd)}")

        # Run the command
        process = subprocess.run(cmd, env=env, capture_output=True, text=True)

        if process.returncode != 0:
            logging.error(f"Marker command failed: {process.stderr}")

        # Process results for each PDF
        for idx, pdf_file in enumerate(pdf_files):
            paper_name = Path(pdf_file).stem
            paper_output_dir = Path(output_dir) / paper_name

            result = {
                "paper_name": paper_name,
                "original_pdf_path": pdf_file,
                "ocr_status": "failed",
                "ocr_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ocr_result_path": "",
            }

            # Check if output was created
            if (
                paper_output_dir.exists()
                and (paper_output_dir / f"{paper_name}.md").exists()
            ):
                result["ocr_status"] = "success"
                result["ocr_result_path"] = str(paper_output_dir)
                status_tracker["completed_files"] += 1
            else:
                result["ocr_status"] = "failed: output not found"
                status_tracker["failed_files"] += 1

            results.append(result)
            global_results.append(result)

            # Update status tracker
            status_tracker["current_time"] = datetime.now()
            status_tracker["time_cost"] = (
                status_tracker["current_time"] - status_tracker["start_time"]
            ).total_seconds()

            # Save to CSV every 50 files
            if len(global_results) % 50 == 0:
                logging.info(f"Saving progress: {len(global_results)} files processed")
                SaveResultsToCsv(global_results, csv_save_path)
                global_results = []  # Clear to avoid duplicate saves

    except Exception as e:
        logging.error(f"Failed to run marker: {str(e)}")
        # Mark all as failed
        for pdf_file in pdf_files:
            result = {
                "paper_name": Path(pdf_file).stem,
                "original_pdf_path": pdf_file,
                "ocr_status": f"failed: {str(e)}",
                "ocr_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ocr_result_path": "",
            }
            results.append(result)
            global_results.append(result)
            status_tracker["failed_files"] += 1

    finally:
        # Cleanup temp directory
        if temp_input_dir.exists():
            shutil.rmtree(temp_input_dir)

    return results


def ProcessPdfSingle(pdf_path: str, output_dir: str) -> Dict:
    """
    Process a single PDF using marker CLI.
    """
    result = {
        "paper_name": Path(pdf_path).stem,
        "original_pdf_path": pdf_path,
        "ocr_status": "failed",
        "ocr_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ocr_result_path": "",
    }

    paper_name = Path(pdf_path).stem
    paper_output_dir = Path(output_dir) / paper_name

    try:
        # Run marker_single command
        cmd = [
            "marker_single",
            str(pdf_path),
            str(paper_output_dir),
            "--output_format",
            "markdown",
        ]

        process = subprocess.run(cmd, capture_output=True, text=True)

        if process.returncode == 0 and paper_output_dir.exists():
            result["ocr_status"] = "success"
            result["ocr_result_path"] = str(paper_output_dir)
        else:
            result["ocr_status"] = f"failed: {process.stderr}"

    except Exception as e:
        result["ocr_status"] = f"failed: {str(e)}"

    return result


def TaskScheduler(
    input_files: List[str], device_workers: Dict[int, int], output_dir: str
) -> List[Dict]:
    """
    Schedule the tasks to the devices using marker CLI.
    """
    global status_tracker, global_results, csv_save_path
    status_tracker["total_files"] = len(input_files)
    status_tracker["start_time"] = datetime.now()

    # Set CSV path
    csv_save_path = Path(output_dir) / "ocr_status.csv"

    gpu_count = len(device_workers)
    total_workers = sum(device_workers.values())

    if gpu_count > 1 and len(input_files) > 1:
        # Use batch mode for multiple GPUs and files
        avg_workers = total_workers // max(1, gpu_count)
        results = ProcessPdfBatch(input_files, output_dir, gpu_count, avg_workers)
    else:
        # Process files one by one for single GPU or single file
        results = []
        for idx, pdf_file in enumerate(input_files):
            result = ProcessPdfSingle(pdf_file, output_dir)
            results.append(result)
            global_results.append(result)

            # Update status
            if result["ocr_status"] == "success":
                status_tracker["completed_files"] += 1
            else:
                status_tracker["failed_files"] += 1

            status_tracker["current_time"] = datetime.now()
            status_tracker["time_cost"] = (
                status_tracker["current_time"] - status_tracker["start_time"]
            ).total_seconds()

            # Save to CSV every 50 files
            if len(global_results) % 50 == 0:
                logging.info(f"Saving progress: {len(global_results)} files processed")
                SaveResultsToCsv(global_results, csv_save_path)
                global_results = []  # Clear to avoid duplicate saves

    # Save any remaining results
    if global_results:
        SaveResultsToCsv(global_results, csv_save_path, append=True)
        global_results = []

    return results


class StatusHandler(BaseHTTPRequestHandler):
    """
    HTTP handler for status tracking.
    """

    def do_GET(self):
        if self.path == "/status":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()

            status = {
                "task_name": status_tracker["task_name"],
                "total_files": status_tracker["total_files"],
                "completed_files": status_tracker["completed_files"],
                "failed_files": status_tracker["failed_files"],
                "time": (
                    status_tracker["current_time"].strftime("%Y-%m-%d %H:%M:%S")
                    if status_tracker["current_time"]
                    else "Not started"
                ),
                "time_cost": status_tracker["time_cost"],
                "progress_percentage": (
                    round(
                        (
                            status_tracker["completed_files"]
                            + status_tracker["failed_files"]
                        )
                        / status_tracker["total_files"]
                        * 100,
                        2,
                    )
                    if status_tracker["total_files"] > 0
                    else 0
                ),
            }

            self.wfile.write(json.dumps(status, indent=2).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress default logging
        pass


def StartStatusServer(port: int = 9090):
    """
    Start HTTP server for status tracking in a separate thread.
    """
    server = HTTPServer(("localhost", port), StatusHandler)
    thread = Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    logging.info(f"Status server started at http://localhost:{port}/status")
    return server


def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load configuration
    config.load()

    # Validate configuration
    if not config.validate():
        logging.error("Configuration validation failed")
        return

    # Create directories
    input_dir = Path(config.paths.input_dir)
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect input files
    logging.info(f"Collecting PDF files from {input_dir}")
    input_files = [str(input_dir / f) for f in CollectInputFiles(str(input_dir))]

    if not input_files:
        logging.warning("No PDF files found in input directory")
        return

    logging.info(f"Found {len(input_files)} PDF files")

    # Detect devices
    gpu_count, device_workers = DeviceDetection()
    total_workers = sum(device_workers.values())
    logging.info(f"Using {total_workers} workers across {len(device_workers)} devices")

    # Start status server
    status_server = StartStatusServer(config.monitoring.metrics_port)

    # Process PDFs
    start_time = time.time()
    results = TaskScheduler(input_files, device_workers, str(output_dir))
    end_time = time.time()

    # Final save (in case there are any remaining)
    csv_path = output_dir / "ocr_status.csv"
    if global_results:
        SaveResultsToCsv(global_results, str(csv_path), append=True)

    logging.info(f"All results saved to {csv_path}")

    # Print summary
    successful = sum(1 for r in results if r["ocr_status"] == "success")
    failed = len(results) - successful

    print("\n" + "=" * 50)
    print("OCR Processing Summary")
    print("=" * 50)
    print(f"Total files: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(
        f"Average time per file: {(end_time - start_time) / len(results):.2f} seconds"
    )
    print(
        f"\nStatus available at: http://localhost:{config.monitoring.metrics_port}/status"
    )
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
