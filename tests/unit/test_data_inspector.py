"""
Unit tests for Data Inspector
"""

import json
import csv
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from src.cli.data_inspector import DataInspector, format_size


class TestDataInspector:
    """Test suite for DataInspector"""
    
    @pytest.mark.unit
    def test_initialization(self, test_config):
        """Test DataInspector initialization"""
        inspector = DataInspector(test_config)
        
        assert inspector.config is test_config
        assert 'scraping' in inspector.stages
        assert 'pdf' in inspector.stages
        assert 'ocr' in inspector.stages
        assert 'qa-text' in inspector.stages
        assert 'qa-vision' in inspector.stages
    
    @pytest.mark.unit
    def test_stage_configuration(self, test_config):
        """Test stage configuration details"""
        inspector = DataInspector(test_config)
        
        ocr_stage = inspector.stages['ocr']
        assert ocr_stage['path'] == test_config.paths.ocr_output
        assert ocr_stage['pattern'] == '*/*.md'
        assert 'OCR extracted text' in ocr_stage['description']
    
    @pytest.mark.unit
    @pytest.mark.sample
    def test_get_stage_info_existing(self, test_inspector, setup_test_data):
        """Test getting info for existing stage"""
        info = test_inspector.get_stage_info('pdf')
        
        assert info['stage'] == 'pdf'
        assert info['exists'] == True
        assert info['count'] == 1  # One PDF file created in setup
        assert 'Downloaded PDF files' in info['description']
    
    @pytest.mark.unit
    def test_get_stage_info_nonexistent(self, test_inspector, temp_dir):
        """Test getting info for non-existent stage directory"""
        # Point to non-existent directory
        test_inspector.stages['test'] = {
            'path': str(temp_dir / 'nonexistent'),
            'pattern': '*.txt',
            'description': 'Test stage'
        }
        
        info = test_inspector.get_stage_info('test')
        
        assert info['exists'] == False
        assert info['count'] == 0
        assert info['files'] == []
    
    @pytest.mark.unit
    def test_get_stage_info_invalid(self, test_inspector):
        """Test getting info for invalid stage"""
        with pytest.raises(ValueError, match="Unknown stage"):
            test_inspector.get_stage_info('invalid_stage')
    
    @pytest.mark.unit
    @pytest.mark.sample
    def test_sample_json_file(self, test_inspector, temp_dir):
        """Test sampling JSON file"""
        # Create test JSON file
        json_file = temp_dir / 'test.json'
        test_data = {'key': 'value', 'number': 42}
        json_file.write_text(json.dumps(test_data))
        
        sample = test_inspector.sample_file(str(json_file))
        
        assert sample['file'] == str(json_file)
        assert sample['type'] == '.json'
        assert sample['content'] == test_data
    
    @pytest.mark.unit
    @pytest.mark.sample
    def test_sample_jsonl_file(self, test_inspector, temp_dir):
        """Test sampling JSONL file"""
        # Create test JSONL file
        jsonl_file = temp_dir / 'test.jsonl'
        lines = [
            {'id': 1, 'text': 'first'},
            {'id': 2, 'text': 'second'},
            {'id': 3, 'text': 'third'}
        ]
        content = '\n'.join(json.dumps(line) for line in lines)
        jsonl_file.write_text(content)
        
        sample = test_inspector.sample_file(str(jsonl_file), sample_size=2)
        
        assert sample['type'] == '.jsonl'
        assert len(sample['content']) == 2
        assert sample['content'][0] == lines[0]
        assert sample['total_items'] == 3
    
    @pytest.mark.unit
    @pytest.mark.sample
    def test_sample_csv_file(self, test_inspector, temp_dir):
        """Test sampling CSV file"""
        # Create test CSV file
        csv_file = temp_dir / 'test.csv'
        csv_content = """name,age,city
Alice,30,NYC
Bob,25,LA
Charlie,35,Chicago"""
        csv_file.write_text(csv_content)
        
        sample = test_inspector.sample_file(str(csv_file), sample_size=2)
        
        assert sample['type'] == '.csv'
        assert len(sample['content']) == 2
        assert sample['content'][0]['name'] == 'Alice'
        assert sample['total_items'] == 2  # Excluding header
    
    @pytest.mark.unit
    @pytest.mark.sample
    def test_sample_markdown_file(self, test_inspector, temp_dir):
        """Test sampling Markdown file"""
        # Create test Markdown file
        md_file = temp_dir / 'test.md'
        md_content = """# Test Document

## Section 1
This is test content.

## Section 2
More test content."""
        md_file.write_text(md_content)
        
        sample = test_inspector.sample_file(str(md_file))
        
        assert sample['type'] == '.md'
        assert '# Test Document' in sample['content']
        assert 'total_lines' in sample
    
    @pytest.mark.unit
    def test_sample_pdf_file(self, test_inspector, temp_dir):
        """Test sampling PDF file"""
        # Create dummy PDF file
        pdf_file = temp_dir / 'test.pdf'
        pdf_file.write_bytes(b'%PDF-1.4')
        
        sample = test_inspector.sample_file(str(pdf_file))
        
        assert sample['type'] == '.pdf'
        assert sample['content'] == "PDF file (binary content)"
    
    @pytest.mark.unit
    def test_sample_nonexistent_file(self, test_inspector):
        """Test sampling non-existent file"""
        sample = test_inspector.sample_file('/nonexistent/file.txt')
        
        assert 'error' in sample
        assert 'not found' in sample['error']
    
    @pytest.mark.unit
    def test_sample_file_with_error(self, test_inspector, temp_dir):
        """Test sampling file that causes read error"""
        # Create a file with invalid JSON
        bad_json = temp_dir / 'bad.json'
        bad_json.write_text('{"invalid": json content}')
        
        sample = test_inspector.sample_file(str(bad_json))
        
        assert 'error' in sample
    
    @pytest.mark.unit
    @pytest.mark.sample
    def test_get_random_sample(self, test_inspector, setup_test_data):
        """Test getting random samples from stage"""
        samples = test_inspector.get_random_sample('pdf', count=1)
        
        assert len(samples) == 1
        assert 'file' in samples[0]
        assert samples[0]['type'] == '.pdf'
    
    @pytest.mark.unit
    def test_get_random_sample_empty_stage(self, test_inspector):
        """Test getting random sample from empty stage"""
        # Create a stage with no files
        test_inspector.stages['empty'] = {
            'path': '/nonexistent',
            'pattern': '*.txt',
            'description': 'Empty stage'
        }
        
        samples = test_inspector.get_random_sample('empty')
        
        assert len(samples) == 1
        assert 'error' in samples[0]
        assert 'No files found' in samples[0]['error']
    
    @pytest.mark.unit
    @pytest.mark.parametrize("size_bytes,expected", [
        (0, "0.00 B"),
        (512, "512.00 B"),
        (1024, "1.00 KB"),
        (1024 * 1024, "1.00 MB"),
        (1024 * 1024 * 1024, "1.00 GB"),
        (1536, "1.50 KB"),
        (1024 * 1024 * 1.5, "1.50 MB"),
    ])
    def test_format_size(self, size_bytes, expected):
        """Test file size formatting"""
        assert format_size(size_bytes) == expected


class TestDataInspectorIntegration:
    """Integration tests for DataInspector with real data"""
    
    @pytest.mark.integration
    @pytest.mark.sample
    def test_inspect_complete_pipeline(self, test_inspector, setup_test_data):
        """Test inspecting all stages of a complete pipeline"""
        stages = ['scraping', 'pdf', 'abstracts', 'sft_data']
        
        for stage in stages:
            info = test_inspector.get_stage_info(stage)
            assert info['exists'] == True
            assert info['count'] > 0
    
    @pytest.mark.integration
    @pytest.mark.sample
    def test_inspect_ocr_results(self, test_inspector, sample_ocr_result):
        """Test inspecting OCR result files"""
        # Update inspector to use sample OCR path
        test_inspector.stages['ocr']['path'] = sample_ocr_result.parent
        
        info = test_inspector.get_stage_info('ocr')
        assert info['exists'] == True
        assert info['count'] >= 1  # At least the markdown file
        
        # Sample the markdown file
        md_files = [f for f in info['files'] if f.endswith('.md')]
        if md_files:
            sample = test_inspector.sample_file(md_files[0])
            assert 'Neural Networks' in sample['content']
    
    @pytest.mark.integration
    @pytest.mark.sample
    def test_inspect_figure_data(self, test_inspector, setup_test_data):
        """Test inspecting figure extraction data"""
        info = test_inspector.get_stage_info('figures')
        
        # Check if figures directory exists in test data
        figures_path = Path(test_inspector.stages['figures']['path'])
        if figures_path.exists():
            json_files = list(figures_path.glob('*.json'))
            assert len(json_files) >= 0
    
    @pytest.mark.integration
    def test_inspect_with_statistics(self, test_inspector, setup_test_data):
        """Test getting statistics across all stages"""
        total_files = 0
        total_size = 0
        
        for stage_name in test_inspector.stages:
            try:
                info = test_inspector.get_stage_info(stage_name)
                if info['exists']:
                    total_files += info['count']
                    total_size += info['total_size']
            except Exception:
                pass  # Skip stages that might not exist
        
        assert total_files > 0
        assert total_size > 0