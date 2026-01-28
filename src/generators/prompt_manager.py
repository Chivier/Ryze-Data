"""Prompt template loading and formatting for QA generation."""

import logging
from pathlib import Path
from typing import Dict, Optional


class PromptManager:
    """Manages loading and formatting of prompt templates."""

    def __init__(self, prompts_dir: str = "./prompts"):
        """Initialize the prompt manager.

        Args:
            prompts_dir: Base directory containing prompt templates.
        """
        self.prompts_dir = Path(prompts_dir)
        self.text_dir = self.prompts_dir / "text"
        self.vision_dir = self.prompts_dir / "vision"
        self._cache: Dict[str, str] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

        if not self.prompts_dir.exists():
            raise ValueError(f"Prompts directory not found: {self.prompts_dir}")

    def _load_template(self, template_path: Path) -> str:
        """Load a template file from disk or cache.

        Args:
            template_path: Path to template file.

        Returns:
            Template content as string.

        Raises:
            FileNotFoundError: If template file doesn't exist.
        """
        cache_key = str(template_path)

        if cache_key not in self._cache:
            if not template_path.exists():
                raise FileNotFoundError(f"Template not found: {template_path}")

            with open(template_path, "r", encoding="utf-8") as f:
                self._cache[cache_key] = f.read()

            self.logger.debug(f"Loaded template: {template_path}")

        return self._cache[cache_key]

    def get_text_prompt(self, prompt_type: str, context: str, **kwargs) -> str:
        """Get a formatted text QA prompt.

        Args:
            prompt_type: Type of prompt (factual, mechanism, application).
            context: Research text to include in prompt.
            **kwargs: Additional format parameters.

        Returns:
            Formatted prompt string.
        """
        template_path = self.text_dir / f"{prompt_type}.txt"
        template = self._load_template(template_path)

        return template.format(context=context, **kwargs)

    def get_vision_prompt(self, prompt_type: str, context: str, **kwargs) -> str:
        """Get a formatted vision QA prompt.

        Args:
            prompt_type: Type of prompt (visual-factual, visual-mechanism, etc.).
            context: Context about the figure to include.
            **kwargs: Additional format parameters.

        Returns:
            Formatted prompt string.
        """
        template_path = self.vision_dir / f"{prompt_type}.txt"
        template = self._load_template(template_path)

        return template.format(context=context, **kwargs)

    def get_quality_prompt(
        self,
        question: str,
        answer: str,
        context: str,
        is_vision: bool = False,
    ) -> str:
        """Get a quality evaluation prompt.

        Args:
            question: The question to evaluate.
            answer: The answer to evaluate.
            context: Original context for the QA pair.
            is_vision: Whether this is for vision QA (uses different template).

        Returns:
            Formatted quality evaluation prompt.
        """
        if is_vision:
            template_path = self.vision_dir / "visual-quality.txt"
        else:
            template_path = self.text_dir / "quality.txt"

        template = self._load_template(template_path)

        return template.format(question=question, answer=answer, context=context)

    def list_text_prompts(self) -> list:
        """List available text prompt types.

        Returns:
            List of prompt type names (without .txt extension).
        """
        if not self.text_dir.exists():
            return []

        return [p.stem for p in self.text_dir.glob("*.txt")]

    def list_vision_prompts(self) -> list:
        """List available vision prompt types.

        Returns:
            List of prompt type names (without .txt extension).
        """
        if not self.vision_dir.exists():
            return []

        return [p.stem for p in self.vision_dir.glob("*.txt")]

    def get_prompt_info(
        self, prompt_type: str, is_vision: bool = False
    ) -> Optional[str]:
        """Get the first line/description of a prompt template.

        Args:
            prompt_type: Type of prompt.
            is_vision: Whether this is a vision prompt.

        Returns:
            First line of template or None if not found.
        """
        try:
            if is_vision:
                template = self._load_template(self.vision_dir / f"{prompt_type}.txt")
            else:
                template = self._load_template(self.text_dir / f"{prompt_type}.txt")

            return template.split("\n")[0]
        except FileNotFoundError:
            return None

    def clear_cache(self) -> None:
        """Clear the template cache."""
        self._cache.clear()
        self.logger.debug("Template cache cleared")
