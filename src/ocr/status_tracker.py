"""OCR status tracking via CSV persistence and HTTP status server."""

import csv
import json
import logging
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Dict, List

from src.ocr.base_ocr import OCRResult

logger = logging.getLogger(__name__)


class OCRStatusTracker:
    """Tracks OCR processing status with CSV persistence and HTTP endpoint.

    Args:
        output_dir: Directory for the status CSV file.
        task_name: Name of the current OCR task.
        metrics_port: Port for the HTTP status server.
    """

    def __init__(
        self,
        output_dir: str,
        task_name: str = "ocr_processing",
        metrics_port: int = 9090,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "ocr_status.csv"
        self.task_name = task_name
        self.metrics_port = metrics_port

        self.total_files: int = 0
        self.completed_files: int = 0
        self.failed_files: int = 0
        self.start_time: datetime = datetime.now()
        self.current_time: datetime = datetime.now()

        self._pending_results: List[Dict[str, str]] = []
        self._server: HTTPServer | None = None
        self._save_interval: int = 50

    @property
    def time_cost(self) -> float:
        """Elapsed time in seconds since tracking started."""
        return (self.current_time - self.start_time).total_seconds()

    @property
    def progress_percentage(self) -> float:
        """Percentage of files processed (completed + failed)."""
        if self.total_files == 0:
            return 0.0
        processed = self.completed_files + self.failed_files
        return round(processed / self.total_files * 100, 2)

    def record_result(self, result: OCRResult) -> None:
        """Record a single OCR result and update counters.

        Args:
            result: The OCR processing result to record.
        """
        if result.ocr_status == "success":
            self.completed_files += 1
        else:
            self.failed_files += 1

        self.current_time = datetime.now()
        self._pending_results.append(result.to_dict())

        if len(self._pending_results) >= self._save_interval:
            self.flush()

    def flush(self) -> None:
        """Write pending results to CSV and clear the buffer."""
        if not self._pending_results:
            return

        append = self.csv_path.exists()
        self._write_csv(self._pending_results, append=append)
        logger.info(
            f"Saved {len(self._pending_results)} results "
            f"(total: {self.completed_files + self.failed_files}/"
            f"{self.total_files})"
        )
        self._pending_results.clear()

    def _write_csv(
        self,
        results: List[Dict[str, str]],
        append: bool = False,
    ) -> None:
        """Write results to CSV file.

        Args:
            results: List of result dicts to write.
            append: Whether to append to existing file.
        """
        mode = "a" if append else "w"
        write_header = not append

        fieldnames = [
            "paper_name",
            "original_pdf_path",
            "ocr_status",
            "ocr_time",
            "ocr_result_path",
        ]

        with open(self.csv_path, mode, newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(results)

    def get_status_dict(self) -> Dict[str, object]:
        """Get current status as a dictionary (for HTTP endpoint).

        Returns:
            Status dictionary with progress information.
        """
        return {
            "task_name": self.task_name,
            "total_files": self.total_files,
            "completed_files": self.completed_files,
            "failed_files": self.failed_files,
            "time": self.current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "time_cost": self.time_cost,
            "progress_percentage": self.progress_percentage,
        }

    def start_server(self) -> None:
        """Start the HTTP status server in a daemon thread."""
        tracker = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/status":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    body = json.dumps(tracker.get_status_dict(), indent=2)
                    self.wfile.write(body.encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass  # Suppress default HTTP logging

        self._server = HTTPServer(("localhost", self.metrics_port), _Handler)
        thread = Thread(target=self._server.serve_forever)
        thread.daemon = True
        thread.start()
        logger.info(
            f"Status server started at " f"http://localhost:{self.metrics_port}/status"
        )

    def stop_server(self) -> None:
        """Stop the HTTP status server if running."""
        if self._server:
            self._server.shutdown()
            self._server = None
