import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class OCRConfig:
    model: str = "marker"
    batch_size: int = 10
    confidence_threshold: float = 0.8
    language: str = "en"
    gpu_enabled: bool = True
    gpu_memory_limit: float = 0.5
    timeout_seconds: int = 300


@dataclass
class PathConfig:
    input_dir: str = "./input"
    output_dir: str = "./output"
    temp_dir: str = "./temp"
    logs_dir: str = "./logs"


@dataclass
class ProcessingConfig:
    parallel_workers: int = 4
    max_retries: int = 3
    retry_delay_seconds: int = 5
    quality_threshold: float = 0.85


@dataclass
class QATemplateConfig:
    version: str = "v1.2"
    templates_dir: str = "./templates"
    enabled_types: list = field(default_factory=lambda: ["factual", "conceptual", "visual", "reference"])


@dataclass
class ModelConfig:
    provider: str = "openai"
    model: str = "gpt-4"
    api_endpoint: str = "https://api.openai.com/v1"
    api_key_env: str = ""
    api_key: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    batch_inference: bool = False
    batch_size: int = 20

    def __post_init__(self):
        if self.api_key_env and not self.api_key:
            self.api_key = os.getenv(self.api_key_env)


@dataclass
class OutputFormatConfig:
    markdown: bool = True
    json: bool = True
    bibtex: bool = True
    include_images: bool = True
    compress_images: bool = False


@dataclass
class MonitoringConfig:
    enable_metrics: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"
    log_format: str = "json"


@dataclass
class ProjectConfig:
    name: str = "Ryze-Data"
    version: str = "1.0.0"
    environment: str = "development"


class ConfigManager:
    _instance = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.config_path = None
            self.project: ProjectConfig = ProjectConfig()
            self.ocr: OCRConfig = OCRConfig()
            self.paths: PathConfig = PathConfig()
            self.processing: ProcessingConfig = ProcessingConfig()
            self.qa_templates: QATemplateConfig = QATemplateConfig()
            self.parsing_model: ModelConfig = ModelConfig()
            self.qa_generation_model: ModelConfig = ModelConfig()
            self.output_formats: OutputFormatConfig = OutputFormatConfig()
            self.monitoring: MonitoringConfig = MonitoringConfig()
    
    def load(self, config_path: str = "config.json") -> None:
        """Load configuration from JSON file and environment variables."""
        self.config_path = Path(config_path)
        
        # Load from JSON file if exists
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self._config = json.load(f)
            
            # Update dataclass instances with loaded config
            self._update_from_dict()
        else:
            # Use defaults if config file doesn't exist
            print(f"Warning: Config file {config_path} not found. Using default values.")
        
        # Override with environment variables
        self._load_env_overrides()
    
    def _update_from_dict(self) -> None:
        """Update dataclass instances from loaded config dictionary."""
        if "project" in self._config:
            self.project = ProjectConfig(**self._config["project"])
        
        if "ocr" in self._config:
            self.ocr = OCRConfig(**self._config["ocr"])
        
        if "paths" in self._config:
            self.paths = PathConfig(**self._config["paths"])
        
        if "processing" in self._config:
            self.processing = ProcessingConfig(**self._config["processing"])
        
        if "qa_templates" in self._config:
            self.qa_templates = QATemplateConfig(**self._config["qa_templates"])
        
        if "parsing_model" in self._config:
            self.parsing_model = ModelConfig(**self._config["parsing_model"])
        
        if "qa_generation_model" in self._config:
            self.qa_generation_model = ModelConfig(**self._config["qa_generation_model"])
        
        if "output_formats" in self._config:
            self.output_formats = OutputFormatConfig(**self._config["output_formats"])
        
        if "monitoring" in self._config:
            self.monitoring = MonitoringConfig(**self._config["monitoring"])
    
    def _load_env_overrides(self) -> None:
        """Load environment variable overrides for sensitive data."""
        # Load API keys from environment
        if self.parsing_model.api_key_env:
            api_key = os.getenv(self.parsing_model.api_key_env)
            if api_key:
                self.parsing_model.api_key = api_key
        
        if self.qa_generation_model.api_key_env:
            api_key = os.getenv(self.qa_generation_model.api_key_env)
            if api_key:
                self.qa_generation_model.api_key = api_key
        
        # Override other configs from environment if needed
        # Format: RYZE_<SECTION>_<KEY> = value
        for key, value in os.environ.items():
            if key.startswith("RYZE_") and not key.endswith("_API_KEY"):
                self._apply_env_override(key, value)
    
    def _apply_env_override(self, env_key: str, value: str) -> None:
        """Apply environment variable override to config."""
        parts = env_key.lower().split("_")
        if len(parts) < 3 or parts[0] != "ryze":
            return
        
        section = parts[1]
        config_key = "_".join(parts[2:])
        
        # Map section to config attribute
        section_map = {
            "ocr": self.ocr,
            "paths": self.paths,
            "processing": self.processing,
            "qa": self.qa_templates,
            "parsing": self.parsing_model,
            "generation": self.qa_generation_model,
            "output": self.output_formats,
            "monitoring": self.monitoring
        }
        
        if section in section_map and hasattr(section_map[section], config_key):
            # Convert value to appropriate type
            current_value = getattr(section_map[section], config_key)
            if isinstance(current_value, bool):
                converted_value = value.lower() in ("true", "1", "yes")
            elif isinstance(current_value, int):
                converted_value = int(value)
            elif isinstance(current_value, float):
                converted_value = float(value)
            else:
                converted_value = value
            
            setattr(section_map[section], config_key, converted_value)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a config value by dot-separated path (e.g., 'ocr.model')."""
        keys = key_path.split(".")
        value = self
        
        for key in keys:
            if hasattr(value, key):
                value = getattr(value, key)
            else:
                return default
        
        return value
    
    def save(self, config_path: Optional[str] = None) -> None:
        """Save current configuration to JSON file."""
        save_path = config_path or self.config_path or "config.json"
        
        config_dict = {
            "project": {
                "name": self.project.name,
                "version": self.project.version,
                "environment": self.project.environment
            },
            "ocr": {
                "model": self.ocr.model,
                "batch_size": self.ocr.batch_size,
                "confidence_threshold": self.ocr.confidence_threshold,
                "language": self.ocr.language,
                "gpu_enabled": self.ocr.gpu_enabled,
                "gpu_memory_limit": self.ocr.gpu_memory_limit,
                "timeout_seconds": self.ocr.timeout_seconds
            },
            "paths": {
                "input_dir": self.paths.input_dir,
                "output_dir": self.paths.output_dir,
                "temp_dir": self.paths.temp_dir,
                "logs_dir": self.paths.logs_dir
            },
            "processing": {
                "parallel_workers": self.processing.parallel_workers,
                "max_retries": self.processing.max_retries,
                "retry_delay_seconds": self.processing.retry_delay_seconds,
                "quality_threshold": self.processing.quality_threshold
            },
            "qa_templates": {
                "version": self.qa_templates.version,
                "templates_dir": self.qa_templates.templates_dir,
                "enabled_types": self.qa_templates.enabled_types
            },
            "parsing_model": {
                "provider": self.parsing_model.provider,
                "model": self.parsing_model.model,
                "api_endpoint": self.parsing_model.api_endpoint,
                "api_key_env": self.parsing_model.api_key_env,
                "max_tokens": self.parsing_model.max_tokens,
                "temperature": self.parsing_model.temperature
            },
            "qa_generation_model": {
                "provider": self.qa_generation_model.provider,
                "model": self.qa_generation_model.model,
                "api_endpoint": self.qa_generation_model.api_endpoint,
                "api_key_env": self.qa_generation_model.api_key_env,
                "max_tokens": self.qa_generation_model.max_tokens,
                "temperature": self.qa_generation_model.temperature,
                "batch_inference": self.qa_generation_model.batch_inference,
                "batch_size": self.qa_generation_model.batch_size
            },
            "output_formats": {
                "markdown": self.output_formats.markdown,
                "json": self.output_formats.json,
                "bibtex": self.output_formats.bibtex,
                "include_images": self.output_formats.include_images,
                "compress_images": self.output_formats.compress_images
            },
            "monitoring": {
                "enable_metrics": self.monitoring.enable_metrics,
                "metrics_port": self.monitoring.metrics_port,
                "log_level": self.monitoring.log_level,
                "log_format": self.monitoring.log_format
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def validate(self) -> bool:
        """Validate the configuration."""
        errors = []
        
        # Check required paths exist or can be created
        for path_attr in ["input_dir", "output_dir", "temp_dir", "logs_dir"]:
            path = getattr(self.paths, path_attr)
            if not Path(path).exists() and path_attr == "input_dir":
                errors.append(f"Input directory {path} does not exist")
        
        # Check API keys are set for models
        if self.parsing_model.api_key_env and not self.parsing_model.api_key:
            errors.append(f"API key not found for parsing model. Set {self.parsing_model.api_key_env} environment variable")
        
        if self.qa_generation_model.api_key_env and not self.qa_generation_model.api_key:
            errors.append(f"API key not found for QA generation model. Set {self.qa_generation_model.api_key_env} environment variable")
        
        # Check value ranges
        if not 0 <= self.ocr.confidence_threshold <= 1:
            errors.append("OCR confidence threshold must be between 0 and 1")
        
        if not 0 <= self.processing.quality_threshold <= 1:
            errors.append("Processing quality threshold must be between 0 and 1")
        
        if self.processing.parallel_workers < 1:
            errors.append("Parallel workers must be at least 1")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True


# Singleton instance
config = ConfigManager()