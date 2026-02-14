"""OCR model registry for discovering and selecting OCR implementations."""

from typing import Dict, List, Optional, Type

from src.ocr.base_ocr import BaseOCRModel


class OCRRegistry:
    """Registry for OCR model classes.

    Provides decorator-based registration and lookup by model name.
    Models are registered as classes (not instances) to avoid
    import-time side effects.

    Usage:
        @OCRRegistry.register
        class MyOCR(BaseOCRModel):
            MODEL_NAME = "my-ocr"
            ...
    """

    _models: Dict[str, Type[BaseOCRModel]] = {}

    @classmethod
    def register(cls, model_class: Type[BaseOCRModel]) -> Type[BaseOCRModel]:
        """Register an OCR model class.

        Args:
            model_class: A BaseOCRModel subclass with MODEL_NAME set.

        Returns:
            The model class unchanged (for use as decorator).

        Raises:
            ValueError: If MODEL_NAME is empty or already registered.
        """
        name = getattr(model_class, "MODEL_NAME", "")
        if not name:
            raise ValueError(
                f"{model_class.__name__} must define a non-empty MODEL_NAME"
            )
        if name in cls._models:
            raise ValueError(
                f"OCR model '{name}' is already registered "
                f"by {cls._models[name].__name__}"
            )
        cls._models[name] = model_class
        return model_class

    @classmethod
    def get_model_class(cls, name: str) -> Optional[Type[BaseOCRModel]]:
        """Look up a registered model class by name.

        Args:
            name: The MODEL_NAME of the desired model.

        Returns:
            The model class, or None if not found.
        """
        return cls._models.get(name)

    @classmethod
    def get_model(cls, name: str, output_dir: str, **kwargs) -> BaseOCRModel:
        """Instantiate a registered model by name.

        Args:
            name: The MODEL_NAME of the desired model.
            output_dir: Directory for OCR output.
            **kwargs: Extra keyword arguments forwarded to the model
                constructor (e.g. ``backend``, ``api_key`` for GLM-OCR,
                or ``mode``, ``device`` for PaddleOCR).

        Returns:
            An instance of the requested OCR model.

        Raises:
            ValueError: If the model is not registered.
            RuntimeError: If the model's dependencies are not available.
        """
        model_class = cls._models.get(name)
        if model_class is None:
            available = ", ".join(cls._models.keys())
            raise ValueError(
                f"Unknown OCR model '{name}'. Registered models: {available}"
            )

        if not model_class.is_available():
            raise RuntimeError(
                f"OCR model '{name}' is registered but its dependencies "
                f"are not installed. Install them and try again."
            )

        return model_class(output_dir=output_dir, **kwargs)

    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered model names.

        Returns:
            Sorted list of model names.
        """
        return sorted(cls._models.keys())

    @classmethod
    def list_available(cls) -> List[str]:
        """List model names whose dependencies are installed.

        Returns:
            Sorted list of available model names.
        """
        return sorted(
            name
            for name, model_class in cls._models.items()
            if model_class.is_available()
        )

    @classmethod
    def list_models_with_status(cls) -> List[Dict[str, str]]:
        """List all models with their availability status.

        Returns:
            List of dicts with 'name' and 'status' keys.
        """
        result = []
        for name in sorted(cls._models.keys()):
            model_class = cls._models[name]
            status = "available" if model_class.is_available() else "not installed"
            result.append({"name": name, "status": status})
        return result

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations. Primarily for testing."""
        cls._models.clear()
