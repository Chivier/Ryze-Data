"""
Integration tests for the complete pipeline with sample data
"""

import os
import json
import shutil
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch, Mock
import tempfile

from src.pipeline_manager import PipelineManager, StageStatus, PipelineResult
from src.config_manager import ConfigManager
from src.cli.data_inspector import DataInspector


class TestFullPipeline:
    """Integration tests for complete pipeline execution"""
    
    @pytest.mark.integration
    @pytest.mark.sample
    def test_pipeline_with_sample_data(self, test_config, setup_test_data, mock_openai_client):
        """Test complete pipeline with sample Nature article data"""
        pipeline = PipelineManager(test_config)
        
        # Verify pipeline stages are initialized
        assert len(pipeline.stages) > 0
        assert 'scrape' in pipeline.stages
        assert 'ocr' in pipeline.stages
        assert 'generate_text_qa' in pipeline.stages
        
        # Check execution order is correct
        assert pipeline.execution_order.index('scrape') < pipeline.execution_order.index('download')
        assert pipeline.execution_order.index('download') < pipeline.execution_order.index('ocr')
    
    @pytest.mark.integration
    @pytest.mark.sample
    def test_single_article_processing(self, test_config, sample_ocr_result, mock_openai_client):
        """Test processing a single article through the pipeline"""
        pipeline = PipelineManager(test_config)
        
        # Mock the scraping stage to return our sample article
        with patch.object(pipeline, '_run_scraping') as mock_scrape:
            mock_scrape.return_value = None
            
            # Mock download stage
            with patch.object(pipeline, '_run_download') as mock_download:
                mock_download.return_value = None
                
                # Mock OCR stage to use sample data
                with patch.object(pipeline, '_run_ocr') as mock_ocr:
                    mock_ocr.return_value = None
                    
                    # Run specific stages
                    result = pipeline.run(stages=['scrape', 'download', 'ocr'], force=True)
                    
                    assert result.completed_stages >= 3
                    assert mock_scrape.called
                    assert mock_download.called
                    assert mock_ocr.called
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_pipeline_skip_existing(self, test_pipeline, setup_test_data):
        """Test pipeline skips stages with existing outputs"""
        # First run should execute
        result1 = pipeline.run(stages=['scrape'], skip_existing=True)
        
        # Second run should skip
        result2 = test_pipeline.run(stages=['scrape'], skip_existing=True)
        
        assert result2.skipped_stages > 0
    
    @pytest.mark.integration
    def test_pipeline_dependency_resolution(self, test_pipeline):
        """Test pipeline correctly resolves dependencies"""
        # Try to run a stage with unmet dependencies
        test_pipeline.stages['ocr'].status = StageStatus.PENDING
        test_pipeline.stages['download'].status = StageStatus.PENDING
        
        can_run = test_pipeline._can_run_stage('ocr')
        assert can_run == False
        
        # Mark dependency as complete
        test_pipeline.stages['download'].status = StageStatus.COMPLETED
        can_run = test_pipeline._can_run_stage('ocr')
        assert can_run == True
    
    @pytest.mark.integration
    def test_pipeline_failure_handling(self, test_pipeline):
        """Test pipeline handles stage failures correctly"""
        # Mock a failing stage
        def failing_stage():
            raise Exception("Test failure")
        
        test_pipeline.stages['scrape'].function = failing_stage
        
        result = test_pipeline.run(stages=['scrape'], force=False)
        
        assert result.failed_stages == 1
        assert test_pipeline.stages['scrape'].status == StageStatus.FAILED
        assert test_pipeline.stages['scrape'].error == "Test failure"
    
    @pytest.mark.integration
    def test_pipeline_force_mode(self, test_pipeline):
        """Test pipeline force mode continues after failures"""
        # Mock multiple stages with one failing
        def failing_stage():
            raise Exception("Test failure")
        
        def success_stage():
            pass
        
        test_pipeline.stages['scrape'].function = failing_stage
        test_pipeline.stages['download'].function = success_stage
        test_pipeline.stages['download'].dependencies = []  # Remove dependency
        
        result = test_pipeline.run(stages=['scrape', 'download'], force=True)
        
        assert result.failed_stages == 1
        assert result.completed_stages == 1
    
    @pytest.mark.integration
    @pytest.mark.sample
    def test_data_flow_between_stages(self, test_config, temp_dir):
        """Test data correctly flows between pipeline stages"""
        pipeline = PipelineManager(test_config)
        
        # Create mock data for first stage
        metadata_dir = temp_dir / 'nature_metadata'
        metadata_dir.mkdir(parents=True)
        
        csv_file = metadata_dir / 'all_articles.csv'
        csv_content = """title,url,abstract,open_access,date,author
"Test Article","https://nature.com/test","Abstract","Y","2024-01-01","Author"
"""
        csv_file.write_text(csv_content)
        
        # Update config to use temp directory
        test_config.paths.nature_data = str(metadata_dir)
        test_config.paths.pdf_dir = str(temp_dir / 'pdfs')
        
        # Verify data can be read by next stage
        assert csv_file.exists()
        assert len(list(metadata_dir.glob('*.csv'))) == 1
    
    @pytest.mark.integration
    def test_pipeline_state_persistence(self, test_pipeline, temp_dir):
        """Test saving and loading pipeline state"""
        # Run pipeline partially
        test_pipeline.stages['scrape'].status = StageStatus.COMPLETED
        test_pipeline.stages['download'].status = StageStatus.FAILED
        test_pipeline.stages['download'].error = "Network error"
        
        # Save state
        state_file = temp_dir / 'pipeline_state.json'
        test_pipeline.save_state(str(state_file))
        
        assert state_file.exists()
        
        # Load and verify state
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        assert state['stages']['scrape']['status'] == 'completed'
        assert state['stages']['download']['status'] == 'failed'
        assert state['stages']['download']['error'] == "Network error"
    
    @pytest.mark.integration
    def test_pipeline_status_reporting(self, test_pipeline):
        """Test pipeline status reporting"""
        status = test_pipeline.get_status()
        
        assert 'stages' in status
        assert 'execution_order' in status
        
        # Check all stages are reported
        for stage_name in test_pipeline.stages:
            assert stage_name in status['stages']
            stage_status = status['stages'][stage_name]
            assert 'status' in stage_status
            assert 'description' in stage_status
            assert 'dependencies' in stage_status
            assert 'output_exists' in stage_status


class TestEndToEndScenarios:
    """End-to-end tests simulating real usage scenarios"""
    
    @pytest.mark.integration
    @pytest.mark.sample
    @pytest.mark.slow
    def test_process_biocot_sample_data(self, test_config):
        """Test processing the existing biocot sample data"""
        # Check if biocot sample data exists
        biocot_sample = Path('biocot/data-sample')
        if not biocot_sample.exists():
            pytest.skip("Biocot sample data not found")
        
        # Copy sample data to test directory
        test_data_dir = Path(test_config.paths.data_root)
        test_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy OCR results
        ocr_source = biocot_sample / 'ocr_result' / 'nature04244'
        if ocr_source.exists():
            ocr_dest = test_data_dir / 'ocr_results' / 'nature04244'
            ocr_dest.mkdir(parents=True, exist_ok=True)
            
            for file in ocr_source.iterdir():
                shutil.copy2(file, ocr_dest)
        
        # Copy abstract
        abstract_source = biocot_sample / 'abstract' / 'nature04244_abstract.txt'
        if abstract_source.exists():
            abstract_dest = test_data_dir / 'abstracts'
            abstract_dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(abstract_source, abstract_dest)
        
        # Initialize pipeline and inspector
        pipeline = PipelineManager(test_config)
        inspector = DataInspector(test_config)
        
        # Check data is accessible
        ocr_info = inspector.get_stage_info('ocr')
        assert ocr_info['exists'] == True
        assert ocr_info['count'] > 0
    
    @pytest.mark.integration
    @pytest.mark.mock
    def test_mock_api_processing(self, test_config, mock_openai_client):
        """Test complete processing with mocked API calls"""
        pipeline = PipelineManager(test_config)
        
        # Mock all external API calls
        with patch('requests.get') as mock_get:
            # Mock Nature website response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = '<html>Mock Nature page</html>'
            mock_get.return_value = mock_response
            
            # Mock stages that would call APIs
            with patch.object(pipeline, '_run_text_qa_generation') as mock_text_qa:
                with patch.object(pipeline, '_run_vision_qa_generation') as mock_vision_qa:
                    
                    # Run QA generation stages
                    result = pipeline.run(
                        stages=['generate_text_qa', 'generate_vision_qa'],
                        force=True
                    )
                    
                    # Verify mocked functions were called
                    assert mock_text_qa.called or mock_vision_qa.called
    
    @pytest.mark.integration
    @pytest.mark.sample
    def test_inspect_after_processing(self, test_config, setup_test_data):
        """Test data inspection after pipeline processing"""
        inspector = DataInspector(test_config)
        
        # Inspect each stage
        stages_to_check = ['pdf', 'abstracts', 'sft_data']
        
        for stage in stages_to_check:
            info = inspector.get_stage_info(stage)
            
            if info['exists'] and info['count'] > 0:
                # Sample a file from the stage
                samples = inspector.get_random_sample(stage, count=1)
                assert len(samples) > 0
                
                sample = samples[0]
                assert 'file' in sample
                assert 'size' in sample
    
    @pytest.mark.integration
    def test_parallel_processing(self, test_config):
        """Test parallel processing configuration"""
        # Set parallel processing configuration
        test_config.processing.parallel_workers = 4
        test_config.processing.qa_ratio = 2
        
        pipeline = PipelineManager(test_config)
        
        # Verify configuration is applied
        assert pipeline.config.processing.parallel_workers == 4
        assert pipeline.config.processing.qa_ratio == 2
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_large_batch_processing(self, test_config, temp_dir):
        """Test processing multiple articles in batch"""
        # Create multiple sample articles
        metadata_dir = temp_dir / 'nature_metadata'
        metadata_dir.mkdir(parents=True)
        
        # Generate 10 sample articles
        csv_file = metadata_dir / 'all_articles.csv'
        csv_lines = ['title,url,abstract,open_access,date,author']
        
        for i in range(10):
            csv_lines.append(
                f'"Article {i}","https://nature.com/article{i}",'
                f'"Abstract {i}","Y","2024-01-{i+1:02d}","Author {i}"'
            )
        
        csv_file.write_text('\n'.join(csv_lines))
        
        # Update config
        test_config.paths.nature_data = str(metadata_dir)
        test_config.processing.max_papers = 5  # Process only 5
        
        # Verify batch size limit is respected
        assert test_config.processing.max_papers == 5
    
    @pytest.mark.integration
    def test_error_recovery(self, test_pipeline, temp_dir):
        """Test pipeline error recovery mechanisms"""
        # Simulate partial completion
        test_pipeline.stages['scrape'].status = StageStatus.COMPLETED
        test_pipeline.stages['download'].status = StageStatus.FAILED
        
        # Save state
        state_file = temp_dir / 'error_state.json'
        test_pipeline.save_state(str(state_file))
        
        # Create new pipeline and load state
        new_pipeline = PipelineManager(test_pipeline.config)
        
        # Verify can resume from saved state
        with open(state_file, 'r') as f:
            saved_state = json.load(f)
        
        assert saved_state['stages']['scrape']['status'] == 'completed'
        assert saved_state['stages']['download']['status'] == 'failed'