"""QA generators for creating training data from scientific papers."""

from src.generators.base_generator import BaseQAGenerator, QAPair
from src.generators.prompt_manager import PromptManager
from src.generators.text_qa_generator import TextQAGenerator
from src.generators.vision_qa_generator import VisionQAGenerator

__all__ = [
    "BaseQAGenerator",
    "QAPair",
    "PromptManager",
    "TextQAGenerator",
    "VisionQAGenerator",
]
