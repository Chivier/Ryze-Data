"""
Unit tests for ConfigManager
"""

import os
import json
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from src.config_manager import ConfigManager, OCRConfig, PathConfig, ProcessingConfig


class TestConfigManager:
    """Test suite for ConfigManager"""
    
    @pytest.mark.unit
    def test_singleton_pattern(self):
        """Test that ConfigManager follows singleton pattern"""
        config1 = ConfigManager()
        config2 = ConfigManager()
        assert config1 is config2
    
    @pytest.mark.unit
    def test_default_initialization(self):
        """Test default configuration values"""
        config = ConfigManager()
        
        assert config.project.name == "Ryze-Data"
        assert config.ocr.model == "marker"
        assert config.ocr.batch_size == 10
        assert config.paths.data_root == "./data"
        assert config.processing.parallel_workers == 4
    
    @pytest.mark.unit
    def test_load_config_file(self, test_config):
        """Test loading configuration from file"""
        assert test_config.project.environment == "test"
        assert test_config.ocr.batch_size == "${RYZE_BATCH_SIZE:2}"
        assert test_config.paths.data_root == "${RYZE_DATA_ROOT:./tests/data}"
    
    @pytest.mark.unit
    def test_environment_variable_expansion(self, temp_dir):
        """Test environment variable expansion in config"""
        # Set environment variable
        os.environ['TEST_VAR'] = 'test_value'
        
        # Create config with env var reference
        config_data = {
            "project": {
                "name": "${TEST_VAR:default}",
                "version": "1.0.0",
                "environment": "test"
            }
        }
        
        config_file = temp_dir / 'test_config.json'
        config_file.write_text(json.dumps(config_data))
        
        config = ConfigManager()
        config.load(str(config_file))
        
        assert config.project.name == "test_value"
    
    @pytest.mark.unit
    def test_environment_override(self):
        """Test environment variable overrides"""
        os.environ['RYZE_OCR_BATCH_SIZE'] = '20'
        os.environ['RYZE_PROCESSING_PARALLEL_WORKERS'] = '8'
        
        config = ConfigManager()
        config.load()
        
        # Environment variables should override defaults
        assert config.ocr.batch_size == 20
        assert config.processing.parallel_workers == 8
    
    @pytest.mark.unit
    def test_api_key_loading(self):
        """Test API key loading from environment"""
        os.environ['RYZE_PARSING_API_KEY'] = 'test-api-key-123'
        
        config = ConfigManager()
        config.parsing_model.api_key_env = 'RYZE_PARSING_API_KEY'
        config._load_env_overrides()
        
        assert config.parsing_model.api_key == 'test-api-key-123'
    
    @pytest.mark.unit
    def test_get_nested_config(self):
        """Test getting nested configuration values"""
        config = ConfigManager()
        
        # Test valid path
        assert config.get('ocr.model') == 'marker'
        assert config.get('paths.data_root') == './data'
        
        # Test invalid path with default
        assert config.get('invalid.path', 'default') == 'default'
    
    @pytest.mark.unit
    def test_save_config(self, temp_dir):
        """Test saving configuration to file"""
        config = ConfigManager()
        save_path = temp_dir / 'saved_config.json'
        
        config.save(str(save_path))
        
        assert save_path.exists()
        
        # Load saved config and verify
        with open(save_path, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data['project']['name'] == 'Ryze-Data'
        assert saved_data['ocr']['model'] == 'marker'
    
    @pytest.mark.unit
    def test_validation_success(self, temp_dir):
        """Test successful configuration validation"""
        config = ConfigManager()
        config.paths.data_root = str(temp_dir)
        config.paths.logs_dir = str(temp_dir / 'logs')
        
        # Set API keys
        config.parsing_model.api_key = 'test-key'
        config.qa_generation_model.api_key = 'test-key'
        
        assert config.validate() == True
    
    @pytest.mark.unit
    def test_validation_failure(self, capsys):
        """Test configuration validation failures"""
        config = ConfigManager()
        
        # Set invalid values
        config.ocr.confidence_threshold = 1.5  # Should be between 0 and 1
        config.processing.parallel_workers = 0  # Should be at least 1
        
        assert config.validate() == False
        
        # Check error messages were printed
        captured = capsys.readouterr()
        assert "OCR confidence threshold must be between 0 and 1" in captured.out
        assert "Parallel workers must be at least 1" in captured.out
    
    @pytest.mark.unit
    def test_dataclass_configs(self):
        """Test individual configuration dataclasses"""
        # Test OCRConfig
        ocr_config = OCRConfig(model="test", batch_size=5)
        assert ocr_config.model == "test"
        assert ocr_config.batch_size == 5
        assert ocr_config.gpu_enabled == True  # Default value
        
        # Test PathConfig
        path_config = PathConfig(data_root="/test/data")
        assert path_config.data_root == "/test/data"
        assert path_config.temp_dir == "./temp"  # Default value
        
        # Test ProcessingConfig
        proc_config = ProcessingConfig(parallel_workers=8, qa_ratio=10)
        assert proc_config.parallel_workers == 8
        assert proc_config.qa_ratio == 10
        assert proc_config.max_retries == 3  # Default value
    
    @pytest.mark.unit
    def test_missing_config_file(self, capsys):
        """Test handling of missing configuration file"""
        config = ConfigManager()
        config.load('nonexistent_config.json')
        
        # Should use defaults and print warning
        captured = capsys.readouterr()
        assert "Config file nonexistent_config.json not found" in captured.out
        assert config.project.name == "Ryze-Data"  # Default value
    
    @pytest.mark.unit
    @pytest.mark.parametrize("env_value,expected_type,expected_value", [
        ("true", bool, True),
        ("false", bool, False),
        ("1", bool, True),
        ("0", bool, False),
        ("42", int, 42),
        ("3.14", float, 3.14),
        ("test_string", str, "test_string"),
    ])
    def test_environment_type_conversion(self, env_value, expected_type, expected_value):
        """Test environment variable type conversion"""
        config = ConfigManager()
        
        # Test boolean conversion
        if expected_type == bool:
            config.ocr.gpu_enabled = False  # Set initial value
            config._apply_env_override("RYZE_OCR_GPU_ENABLED", env_value)
            assert config.ocr.gpu_enabled == expected_value
        
        # Test integer conversion
        elif expected_type == int:
            config.ocr.batch_size = 0  # Set initial value
            config._apply_env_override("RYZE_OCR_BATCH_SIZE", env_value)
            assert config.ocr.batch_size == expected_value
        
        # Test float conversion
        elif expected_type == float:
            config.ocr.confidence_threshold = 0.0  # Set initial value
            config._apply_env_override("RYZE_OCR_CONFIDENCE_THRESHOLD", env_value)
            assert config.ocr.confidence_threshold == expected_value


class TestConfigIntegration:
    """Integration tests for ConfigManager with other components"""
    
    @pytest.mark.integration
    @pytest.mark.sample
    def test_config_with_pipeline(self, test_config):
        """Test configuration integration with pipeline"""
        from src.pipeline_manager import PipelineManager
        
        pipeline = PipelineManager(test_config)
        assert pipeline.config is test_config
        assert len(pipeline.stages) > 0
    
    @pytest.mark.integration
    @pytest.mark.sample  
    def test_config_with_inspector(self, test_config):
        """Test configuration integration with data inspector"""
        from src.cli.data_inspector import DataInspector
        
        inspector = DataInspector(test_config)
        assert inspector.config is test_config
        assert 'ocr' in inspector.stages
    
    @pytest.mark.integration
    def test_complete_config_lifecycle(self, temp_dir):
        """Test complete configuration lifecycle"""
        # Create config
        config = ConfigManager()
        
        # Modify values
        config.project.name = "Test Project"
        config.ocr.batch_size = 15
        config.paths.data_root = str(temp_dir)
        
        # Save config
        save_path = temp_dir / 'lifecycle_config.json'
        config.save(str(save_path))
        
        # Create new instance and load
        new_config = ConfigManager()
        new_config.load(str(save_path))
        
        # Verify values persisted
        assert new_config.project.name == "Test Project"
        assert new_config.ocr.batch_size == 15
        assert new_config.paths.data_root == str(temp_dir)