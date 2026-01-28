"""Vision-based QA generator from figure images."""

import base64
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.api_key_balancer import RequestStatus
from src.generators.base_generator import BaseQAGenerator, QAPair
from src.generators.prompt_manager import PromptManager


class VisionQAGenerator(BaseQAGenerator):
    """Generate QA pairs from scientific figures using vision models."""

    # Vision prompt types and their target question counts
    PROMPT_CONFIG: Dict[str, int] = {
        "visual-factual": 3,
        "visual-mechanism": 2,
        "visual-data-extraction": 1,
        "visual-analysis": 3,
        "visual-comparison": 2,
    }

    def __init__(
        self,
        vlm_dir: str,
        abstract_dir: str,
        output_dir: str,
        model: str,
        workers: int = 4,
        qa_ratio: int = 8,
        quality_filter: bool = False,
        quality_threshold: float = 2.5,
    ):
        """Initialize the vision QA generator.

        Args:
            vlm_dir: Directory containing figure context JSON and images.
            abstract_dir: Directory containing paper abstracts (optional).
            output_dir: Directory to save generated QA pairs.
            model: Vision model name for API calls.
            workers: Number of parallel workers for processing.
            qa_ratio: Target QA pairs per figure (used for sampling).
            quality_filter: Whether to filter by quality score.
            quality_threshold: Minimum quality score if filtering enabled.
        """
        super().__init__(output_dir=output_dir, model=model, qa_ratio=qa_ratio)

        self.vlm_dir = Path(vlm_dir)
        self.abstract_dir = Path(abstract_dir) if abstract_dir else None
        self.workers = workers
        self.quality_filter = quality_filter
        self.quality_threshold = quality_threshold

        # Initialize prompt manager
        prompts_dir = self.config.paths.prompts_dir
        self.prompt_manager = PromptManager(prompts_dir)

    def run(self) -> None:
        """Run the vision QA generation pipeline."""
        # Discover papers with figure context
        context_files = list(self.vlm_dir.glob("*.json"))
        if not context_files:
            self.logger.warning(f"No context JSON files found in {self.vlm_dir}")
            return

        self.logger.info(f"Found {len(context_files)} papers with figures")

        # Initialize API balancer
        api_keys = self._get_api_keys(self.config.vision_model)
        self._init_balancer(api_keys)

        try:
            # Process papers in parallel with thread pool
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {
                    executor.submit(self._process_paper_safe, ctx_file): ctx_file
                    for ctx_file in context_files
                }

                with self.create_progress() as progress:
                    task = progress.add_task(
                        "Processing papers...", total=len(context_files)
                    )

                    for future in as_completed(futures):
                        ctx_file = futures[future]
                        try:
                            qa_pairs = future.result()
                            if qa_pairs:
                                paper_id = ctx_file.stem
                                self._save_vision_qa(qa_pairs, f"{paper_id}_vision_qa")
                        except Exception as e:
                            self._log_error(
                                "paper_processing",
                                str(e),
                                paper=str(ctx_file),
                            )

                        progress.advance(task)

        finally:
            self.shutdown()

    def _process_paper_safe(self, context_file: Path) -> List[QAPair]:
        """Safe wrapper for process_paper with exception handling.

        Args:
            context_file: Path to figure context JSON.

        Returns:
            List of QA pairs or empty list on error.
        """
        try:
            return self.process_paper(context_file)
        except Exception as e:
            self._log_error("process_paper", str(e), file=str(context_file))
            return []

    def process_paper(self, paper_path: Path) -> List[QAPair]:
        """Process a paper's figures and generate vision QA pairs.

        Args:
            paper_path: Path to figure context JSON file.

        Returns:
            List of generated QA pairs.
        """
        paper_id = paper_path.stem
        self.logger.info(f"Processing paper: {paper_id}")

        # Load figure context
        with open(paper_path, "r", encoding="utf-8") as f:
            figure_data = json.load(f)

        # Handle different context formats
        figures = self._extract_figures(figure_data, paper_path.parent)

        if not figures:
            self._log_error("no_figures", "No valid figures found", paper=paper_id)
            return []

        self.logger.info(f"Paper {paper_id}: {len(figures)} figures")

        all_qa_pairs: List[QAPair] = []

        for figure_info in figures:
            qa_pairs = self._generate_figure_qa(
                figure_info=figure_info,
                paper_id=paper_id,
            )
            all_qa_pairs.extend(qa_pairs)

        # Sample down if too many QA pairs
        if len(all_qa_pairs) > self.qa_ratio * len(figures):
            all_qa_pairs = random.sample(all_qa_pairs, self.qa_ratio * len(figures))

        self.logger.info(f"Paper {paper_id}: Generated {len(all_qa_pairs)} QA pairs")
        return all_qa_pairs

    def _extract_figures(
        self, figure_data: Any, base_dir: Path
    ) -> List[Dict[str, Any]]:
        """Extract figure information from context data.

        Args:
            figure_data: Loaded JSON data (may be dict or list).
            base_dir: Base directory for relative image paths.

        Returns:
            List of figure info dicts with 'image_path' and 'context'.
        """
        figures = []

        # Handle list format
        if isinstance(figure_data, list):
            for item in figure_data:
                figure = self._parse_figure_item(item, base_dir)
                if figure:
                    figures.append(figure)

        # Handle dict with 'figures' key
        elif isinstance(figure_data, dict):
            if "figures" in figure_data:
                for item in figure_data["figures"]:
                    figure = self._parse_figure_item(item, base_dir)
                    if figure:
                        figures.append(figure)
            else:
                # Single figure dict
                figure = self._parse_figure_item(figure_data, base_dir)
                if figure:
                    figures.append(figure)

        return figures

    def _parse_figure_item(
        self, item: Dict[str, Any], base_dir: Path
    ) -> Optional[Dict[str, Any]]:
        """Parse a single figure item from context data.

        Args:
            item: Figure data dict.
            base_dir: Base directory for relative paths.

        Returns:
            Parsed figure info or None if invalid.
        """
        # Find image path
        image_path = None
        for key in ["image_path", "path", "figure_path", "file"]:
            if key in item:
                candidate = base_dir / item[key]
                if candidate.exists():
                    image_path = candidate
                    break

        if not image_path:
            return None

        # Build context from available info
        context_parts = []

        # Add caption
        if "caption" in item:
            context_parts.append(f"Caption: {item['caption']}")

        # Add related text
        for key in ["related_info", "related_text", "context"]:
            if key in item and isinstance(item[key], dict):
                info = item[key]
                if "before" in info:
                    context_parts.append(f"Text before figure: {info['before']}")
                if "after" in info:
                    context_parts.append(f"Text after figure: {info['after']}")
            elif key in item and isinstance(item[key], str):
                context_parts.append(item[key])

        # Add figure type/label
        if "label" in item:
            context_parts.insert(0, f"Figure label: {item['label']}")
        if "figure_type" in item:
            context_parts.insert(0, f"Figure type: {item['figure_type']}")

        return {
            "image_path": image_path,
            "context": "\n".join(context_parts) if context_parts else "",
            "figure_id": item.get("id", item.get("label", image_path.stem)),
            "raw_data": item,
        }

    def _generate_figure_qa(
        self,
        figure_info: Dict[str, Any],
        paper_id: str,
    ) -> List[QAPair]:
        """Generate QA pairs for a single figure.

        Args:
            figure_info: Figure information dict.
            paper_id: Paper identifier.

        Returns:
            List of QA pairs for this figure.
        """
        qa_pairs: List[QAPair] = []
        image_path: Path = figure_info["image_path"]
        context = figure_info["context"]
        figure_id = figure_info["figure_id"]

        # Encode image as base64
        try:
            image_b64 = self._encode_image(image_path)
        except Exception as e:
            self._log_error(
                "image_encoding",
                str(e),
                paper=paper_id,
                figure=figure_id,
            )
            return []

        request_map: Dict[str, Tuple[str, str]] = {}  # request_id -> (prompt_type, ctx)

        # Submit requests for all prompt types
        for prompt_type, question_count in self.PROMPT_CONFIG.items():
            try:
                prompt = self.prompt_manager.get_vision_prompt(
                    prompt_type=prompt_type,
                    context=context,
                )

                # Build vision message with image
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ]

                request_id = self.balancer.submit_chat_completion(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2048,
                )

                request_map[request_id] = (prompt_type, context)

            except Exception as e:
                self._log_error(
                    "vision_prompt_submission",
                    str(e),
                    paper=paper_id,
                    figure=figure_id,
                    prompt_type=prompt_type,
                )

        # Wait for results
        pending_ids = set(request_map.keys())
        timeout = 90.0  # Vision requests may take longer
        start_time = time.time()

        while pending_ids and (time.time() - start_time) < timeout:
            result = self.balancer.get_result(timeout=1.0)
            if result is None:
                continue

            if result.id not in request_map:
                self.balancer.result_queue.put(result)
                continue

            pending_ids.discard(result.id)
            prompt_type, ctx = request_map[result.id]

            if result.status == RequestStatus.SUCCESS:
                response_text = result.result.choices[0].message.content
                parsed = self._parse_json_response(response_text)

                if parsed and "questions" in parsed:
                    for qa_data in parsed["questions"]:
                        qa_pair = QAPair(
                            question=qa_data.get("q", ""),
                            answer=qa_data.get("a", ""),
                            difficulty=qa_data.get("difficulty", "medium"),
                            question_type=prompt_type.replace("visual-", ""),
                            paper_id=paper_id,
                            section=figure_id,
                            context=ctx[:500],
                            metadata={
                                "image_path": str(image_path),
                                "figure_id": figure_id,
                            },
                        )
                        qa_pairs.append(qa_pair)
            else:
                self._log_error(
                    "vision_request_failed",
                    str(result.error),
                    paper=paper_id,
                    figure=figure_id,
                    prompt_type=prompt_type,
                )

        if pending_ids:
            self.logger.warning(
                f"Timeout: {len(pending_ids)} pending requests "
                f"(paper={paper_id}, figure={figure_id})"
            )

        return qa_pairs

    def _encode_image(self, image_path: Path) -> str:
        """Encode image file to base64.

        Args:
            image_path: Path to image file.

        Returns:
            Base64 encoded string.
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _save_vision_qa(self, qa_pairs: List[QAPair], filename: str) -> Path:
        """Save vision QA pairs in LlamaFactory-compatible format.

        Args:
            qa_pairs: List of QA pairs with image metadata.
            filename: Output filename (without extension).

        Returns:
            Path to saved file.
        """
        output_path = self.output_dir / f"{filename}.jsonl"

        with open(output_path, "w", encoding="utf-8") as f:
            for qa in qa_pairs:
                # LlamaFactory format with <image> placeholder
                llama_entry = {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{qa.question} <image>",
                        },
                        {
                            "role": "assistant",
                            "content": qa.answer,
                        },
                    ],
                    "images": [qa.metadata.get("image_path", "")],
                    "metadata": {
                        "paper_id": qa.paper_id,
                        "figure_id": qa.metadata.get("figure_id", ""),
                        "question_type": qa.question_type,
                        "difficulty": qa.difficulty,
                    },
                }
                f.write(json.dumps(llama_entry, ensure_ascii=False) + "\n")

        self.logger.info(f"Saved {len(qa_pairs)} vision QA pairs to {output_path}")
        return output_path
