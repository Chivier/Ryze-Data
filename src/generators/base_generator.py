"""Base class for QA generators with shared functionality."""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from src.api_key_balancer import OpenAIAPIBalancer
from src.config_manager import ConfigManager


@dataclass
class QAPair:
    """Represents a single question-answer pair."""

    question: str
    answer: str
    difficulty: str = "medium"
    question_type: str = "factual"
    paper_id: str = ""
    section: str = ""
    context: str = ""
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_jsonl_line(self) -> str:
        """Convert to JSONL format."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class BaseQAGenerator(ABC):
    """Abstract base class for QA generators."""

    def __init__(
        self,
        output_dir: str,
        model: str,
        qa_ratio: int = 8,
        config: Optional[ConfigManager] = None,
    ):
        """Initialize the generator.

        Args:
            output_dir: Directory to save generated QA pairs.
            model: Model name for API calls.
            qa_ratio: Target number of QA pairs per unit (section/figure).
            config: Optional ConfigManager instance.
        """
        self.output_dir = Path(output_dir)
        self.model = model
        self.qa_ratio = qa_ratio
        self.config = config or ConfigManager()
        self.console = Console()
        self.logger = self._setup_logger()
        self.balancer: Optional[OpenAIAPIBalancer] = None
        self._errors: List[Dict[str, Any]] = []

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        """Set up logging for this generator."""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _init_balancer(self, api_keys: List[str]) -> None:
        """Initialize the API balancer with given keys.

        Args:
            api_keys: List of API keys for load balancing.
        """
        if not api_keys:
            raise ValueError("At least one API key is required")

        self.balancer = OpenAIAPIBalancer(api_keys=api_keys)
        self.logger.info(f"Initialized API balancer with {len(api_keys)} keys")

    def _get_api_keys(self, model_config) -> List[str]:
        """Extract API keys from model config or environment.

        Args:
            model_config: Model configuration with api_key or api_key_env.

        Returns:
            List of API keys (may be single key in list).
        """
        import os

        keys = []

        # Check direct api_key
        if model_config.api_key:
            keys.append(model_config.api_key)

        # Check environment variable
        if model_config.api_key_env:
            env_key = os.getenv(model_config.api_key_env)
            if env_key and env_key not in keys:
                keys.append(env_key)

        # Check for multiple keys in env (comma-separated)
        multi_key_env = os.getenv("OPENAI_API_KEYS")
        if multi_key_env:
            for key in multi_key_env.split(","):
                key = key.strip()
                if key and key not in keys:
                    keys.append(key)

        if not keys:
            raise ValueError(
                f"No API keys found. Set {model_config.api_key_env} or OPENAI_API_KEYS"
            )

        return keys

    def _filter_qa_by_quality(
        self, qa_pairs: List[QAPair], threshold: float
    ) -> List[QAPair]:
        """Filter QA pairs by quality score.

        Args:
            qa_pairs: List of QA pairs with quality scores.
            threshold: Minimum quality score to keep.

        Returns:
            Filtered list of QA pairs.
        """
        filtered = [qa for qa in qa_pairs if qa.quality_score >= threshold]
        self.logger.info(
            f"Quality filter: {len(filtered)}/{len(qa_pairs)} pairs above {threshold}"
        )
        return filtered

    def _parse_json_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response, handling common issues.

        Args:
            response_text: Raw response text from LLM.

        Returns:
            Parsed JSON dict or None if parsing fails.
        """
        if not response_text:
            return None

        # Try direct parse first
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code block
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try finding JSON object
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        self.logger.warning(
            f"Failed to parse JSON from response: {response_text[:200]}"
        )
        return None

    def _save_qa_pairs(self, qa_pairs: List[QAPair], filename: str) -> Path:
        """Save QA pairs to JSONL file.

        Args:
            qa_pairs: List of QA pairs to save.
            filename: Output filename (without extension).

        Returns:
            Path to saved file.
        """
        output_path = self.output_dir / f"{filename}.jsonl"

        with open(output_path, "w", encoding="utf-8") as f:
            for qa in qa_pairs:
                f.write(qa.to_jsonl_line() + "\n")

        self.logger.info(f"Saved {len(qa_pairs)} QA pairs to {output_path}")
        return output_path

    def _save_errors(self, filename: str = "errors") -> None:
        """Save accumulated errors to file.

        Args:
            filename: Output filename for errors.
        """
        if not self._errors:
            return

        output_path = self.output_dir / f"{filename}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for error in self._errors:
                f.write(json.dumps(error, ensure_ascii=False) + "\n")

        self.logger.warning(f"Saved {len(self._errors)} errors to {output_path}")

    def _log_error(self, error_type: str, message: str, **kwargs) -> None:
        """Log and store an error.

        Args:
            error_type: Category of error.
            message: Error message.
            **kwargs: Additional error context.
        """
        error = {"type": error_type, "message": message, **kwargs}
        self._errors.append(error)
        self.logger.error(f"{error_type}: {message}")

    def create_progress(self) -> Progress:
        """Create a rich Progress instance for tracking."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )

    @abstractmethod
    def run(self) -> None:
        """Run the QA generation pipeline."""
        pass

    @abstractmethod
    def process_paper(self, paper_path: Path) -> List[QAPair]:
        """Process a single paper and generate QA pairs.

        Args:
            paper_path: Path to paper file.

        Returns:
            List of generated QA pairs.
        """
        pass

    def shutdown(self) -> None:
        """Clean up resources."""
        if self.balancer:
            self.balancer.shutdown(wait=True)
            self.logger.info("API balancer shutdown complete")

        self._save_errors()
