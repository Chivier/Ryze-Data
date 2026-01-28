"""Text-based QA generator from OCR markdown files."""

import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

from src.api_key_balancer import RequestStatus
from src.generators.base_generator import BaseQAGenerator, QAPair
from src.generators.prompt_manager import PromptManager


class TextQAGenerator(BaseQAGenerator):
    """Generate QA pairs from OCR-processed markdown files."""

    # Prompt types and their target question counts
    PROMPT_CONFIG: Dict[str, int] = {
        "factual": 3,
        "mechanism": 2,
        "application": 2,
    }

    def __init__(
        self,
        ocr_dir: str,
        abstract_dir: str,
        output_dir: str,
        model: str,
        qa_ratio: int = 8,
        quality_filter: bool = False,
        quality_threshold: float = 2.5,
        max_section_chars: int = 3000,
    ):
        """Initialize the text QA generator.

        Args:
            ocr_dir: Directory containing OCR markdown files.
            abstract_dir: Directory containing paper abstracts (optional context).
            output_dir: Directory to save generated QA pairs.
            model: Model name for API calls.
            qa_ratio: Target QA pairs per section (used for sampling).
            quality_filter: Whether to filter by quality score.
            quality_threshold: Minimum quality score if filtering enabled.
            max_section_chars: Maximum characters per section chunk.
        """
        super().__init__(output_dir=output_dir, model=model, qa_ratio=qa_ratio)

        self.ocr_dir = Path(ocr_dir)
        self.abstract_dir = Path(abstract_dir) if abstract_dir else None
        self.quality_filter = quality_filter
        self.quality_threshold = quality_threshold
        self.max_section_chars = max_section_chars

        # Initialize prompt manager
        prompts_dir = self.config.paths.prompts_dir
        self.prompt_manager = PromptManager(prompts_dir)

    def run(self) -> None:
        """Run the text QA generation pipeline."""
        # Discover papers
        papers = list(self.ocr_dir.glob("*.md"))
        if not papers:
            self.logger.warning(f"No markdown files found in {self.ocr_dir}")
            return

        self.logger.info(f"Found {len(papers)} papers to process")

        # Initialize API balancer
        api_keys = self._get_api_keys(self.config.qa_generation_model)
        self._init_balancer(api_keys)

        try:
            with self.create_progress() as progress:
                task = progress.add_task("Processing papers...", total=len(papers))

                for paper_path in papers:
                    try:
                        qa_pairs = self.process_paper(paper_path)
                        if qa_pairs:
                            paper_id = paper_path.stem
                            self._save_qa_pairs(qa_pairs, f"{paper_id}_qa")
                    except Exception as e:
                        self._log_error(
                            "paper_processing",
                            str(e),
                            paper=str(paper_path),
                        )

                    progress.advance(task)

        finally:
            self.shutdown()

    def process_paper(self, paper_path: Path) -> List[QAPair]:
        """Process a single paper and generate QA pairs.

        Args:
            paper_path: Path to markdown file.

        Returns:
            List of generated QA pairs.
        """
        paper_id = paper_path.stem
        self.logger.info(f"Processing paper: {paper_id}")

        # Load markdown content
        content = paper_path.read_text(encoding="utf-8")

        # Split into sections
        sections = self._split_into_sections(content)
        if not sections:
            self._log_error("no_sections", "No valid sections found", paper=paper_id)
            return []

        self.logger.info(f"Paper {paper_id}: {len(sections)} sections")

        # Generate QA for each section
        all_qa_pairs: List[QAPair] = []

        for idx, section_text in enumerate(sections):
            section_id = f"section_{idx}"
            qa_pairs = self._generate_section_qa(
                section_text=section_text,
                paper_id=paper_id,
                section_id=section_id,
            )
            all_qa_pairs.extend(qa_pairs)

        # Apply quality filter if enabled
        if self.quality_filter and all_qa_pairs:
            all_qa_pairs = self._apply_quality_filter(all_qa_pairs, paper_id)

        self.logger.info(f"Paper {paper_id}: Generated {len(all_qa_pairs)} QA pairs")
        return all_qa_pairs

    def _split_into_sections(self, content: str) -> List[str]:
        """Split markdown content into sections.

        Args:
            content: Full markdown content.

        Returns:
            List of section texts.
        """
        # Split by level 2 headers (##)
        sections = re.split(r"\n##\s+", content)

        result = []
        for section in sections:
            section = section.strip()
            if not section:
                continue

            # Skip very short sections
            if len(section) < 100:
                continue

            # Chunk if too long
            if len(section) > self.max_section_chars:
                chunks = self._chunk_text(section, self.max_section_chars)
                result.extend(chunks)
            else:
                result.append(section)

        return result

    def _chunk_text(self, text: str, max_chars: int) -> List[str]:
        """Split text into chunks respecting paragraph boundaries.

        Args:
            text: Text to chunk.
            max_chars: Maximum characters per chunk.

        Returns:
            List of text chunks.
        """
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para_len = len(para)

            if current_length + para_len > max_chars and current_chunk:
                # Save current chunk and start new one
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_length = para_len
            else:
                current_chunk.append(para)
                current_length += para_len

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _generate_section_qa(
        self,
        section_text: str,
        paper_id: str,
        section_id: str,
    ) -> List[QAPair]:
        """Generate QA pairs for a single section.

        Args:
            section_text: Section content.
            paper_id: Paper identifier.
            section_id: Section identifier.

        Returns:
            List of QA pairs for this section.
        """
        qa_pairs: List[QAPair] = []
        request_map: Dict[str, Tuple[str, str]] = (
            {}
        )  # request_id -> (prompt_type, text)

        # Submit requests for all prompt types
        for prompt_type, question_count in self.PROMPT_CONFIG.items():
            try:
                prompt = self.prompt_manager.get_text_prompt(
                    prompt_type=prompt_type,
                    context=section_text,
                )

                request_id = self.balancer.submit_chat_completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=2048,
                )

                request_map[request_id] = (prompt_type, section_text)

            except Exception as e:
                self._log_error(
                    "prompt_submission",
                    str(e),
                    paper=paper_id,
                    section=section_id,
                    prompt_type=prompt_type,
                )

        # Wait for all results (with timeout)
        pending_ids = set(request_map.keys())
        timeout = 60.0  # 60 seconds timeout
        start_time = time.time()

        while pending_ids and (time.time() - start_time) < timeout:
            result = self.balancer.get_result(timeout=1.0)
            if result is None:
                continue

            if result.id not in request_map:
                # Result for different request, put back
                self.balancer.result_queue.put(result)
                continue

            pending_ids.discard(result.id)
            prompt_type, context = request_map[result.id]

            if result.status == RequestStatus.SUCCESS:
                # Parse response
                response_text = result.result.choices[0].message.content
                parsed = self._parse_json_response(response_text)

                if parsed and "questions" in parsed:
                    for qa_data in parsed["questions"]:
                        qa_pair = QAPair(
                            question=qa_data.get("q", ""),
                            answer=qa_data.get("a", ""),
                            difficulty=qa_data.get("difficulty", "medium"),
                            question_type=prompt_type,
                            paper_id=paper_id,
                            section=section_id,
                            context=context[:500],  # Truncate context for storage
                        )
                        qa_pairs.append(qa_pair)
            else:
                self._log_error(
                    "api_request_failed",
                    str(result.error),
                    paper=paper_id,
                    section=section_id,
                    prompt_type=prompt_type,
                )

        # Log timeout warnings
        if pending_ids:
            self.logger.warning(
                f"Timeout waiting for {len(pending_ids)} requests "
                f"(paper={paper_id}, section={section_id})"
            )

        return qa_pairs

    def _apply_quality_filter(
        self, qa_pairs: List[QAPair], paper_id: str
    ) -> List[QAPair]:
        """Apply quality filtering to QA pairs using LLM evaluation.

        Args:
            qa_pairs: List of QA pairs to filter.
            paper_id: Paper identifier for logging.

        Returns:
            Filtered list of QA pairs.
        """
        self.logger.info(f"Applying quality filter to {len(qa_pairs)} pairs")

        request_map: Dict[str, QAPair] = {}

        # Submit quality evaluation requests
        for qa in qa_pairs:
            try:
                prompt = self.prompt_manager.get_quality_prompt(
                    question=qa.question,
                    answer=qa.answer,
                    context=qa.context,
                    is_vision=False,
                )

                request_id = self.balancer.submit_chat_completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=256,
                )

                request_map[request_id] = qa

            except Exception as e:
                self._log_error("quality_submission", str(e), paper=paper_id)

        # Collect results
        pending_ids = set(request_map.keys())
        timeout = 120.0
        start_time = time.time()

        while pending_ids and (time.time() - start_time) < timeout:
            result = self.balancer.get_result(timeout=1.0)
            if result is None:
                continue

            if result.id not in request_map:
                self.balancer.result_queue.put(result)
                continue

            pending_ids.discard(result.id)
            qa = request_map[result.id]

            if result.status == RequestStatus.SUCCESS:
                response_text = result.result.choices[0].message.content
                score = self._parse_quality_score(response_text)
                qa.quality_score = score

        # Filter by threshold
        return self._filter_qa_by_quality(qa_pairs, self.quality_threshold)

    def _parse_quality_score(self, response_text: str) -> float:
        """Parse quality score from LLM response.

        Args:
            response_text: Response containing "Rate: X" format.

        Returns:
            Parsed score as float, or 0.0 if parsing fails.
        """
        match = re.search(r"Rate:\s*(\d+(?:\.\d+)?)", response_text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

        return 0.0
