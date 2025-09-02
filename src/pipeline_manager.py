#!/usr/bin/env python3
"""
Pipeline Manager: Orchestrates the complete data processing workflow
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import subprocess
from enum import Enum

from src.config_manager import ConfigManager


class StageStatus(Enum):
    """Pipeline stage execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineStage:
    """Represents a single stage in the pipeline"""
    name: str
    description: str
    function: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    status: StageStatus = StageStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    output_path: Optional[str] = None


@dataclass
class PipelineResult:
    """Results from pipeline execution"""
    total_stages: int
    completed_stages: int
    failed_stages: int
    skipped_stages: int
    total_time: float
    stage_results: Dict[str, Dict[str, Any]]


class PipelineManager:
    """Manages and orchestrates the data processing pipeline"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.stages: Dict[str, PipelineStage] = {}
        self.execution_order: List[str] = []
        self.logger = self._setup_logger()
        
        # Initialize pipeline stages
        self._initialize_stages()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the pipeline"""
        logger = logging.getLogger('PipelineManager')
        logger.setLevel(getattr(logging, self.config.monitoring.log_level))
        
        # Create logs directory if it doesn't exist
        log_dir = Path(self.config.paths.logs_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler
        fh = logging.FileHandler(log_dir / f'pipeline_{datetime.now():%Y%m%d_%H%M%S}.log')
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _initialize_stages(self):
        """Initialize all pipeline stages"""
        
        # Stage 1: Scraping
        self.add_stage(
            name="scrape",
            description="Scrape Nature articles metadata",
            function=self._run_scraping,
            dependencies=[],
            output_path=self.config.paths.nature_data
        )
        
        # Stage 2: Download PDFs
        self.add_stage(
            name="download",
            description="Download PDF files",
            function=self._run_download,
            dependencies=["scrape"],
            output_path=self.config.paths.pdf_dir
        )
        
        # Stage 3: OCR Processing
        self.add_stage(
            name="ocr",
            description="Extract text and images using OCR",
            function=self._run_ocr,
            dependencies=["download"],
            output_path=self.config.paths.ocr_output
        )
        
        # Stage 4: Figure Extraction
        self.add_stage(
            name="extract_figures",
            description="Extract figures and related context",
            function=self._run_figure_extraction,
            dependencies=["ocr"],
            output_path=self.config.paths.vlm_preprocessing
        )
        
        # Stage 5: Abstract Extraction
        self.add_stage(
            name="extract_abstracts",
            description="Extract paper abstracts",
            function=self._run_abstract_extraction,
            dependencies=["ocr"],
            output_path=self.config.paths.abstract_dir
        )
        
        # Stage 6: Text QA Generation
        self.add_stage(
            name="generate_text_qa",
            description="Generate text-based QA pairs",
            function=self._run_text_qa_generation,
            dependencies=["ocr", "extract_abstracts"],
            output_path=self.config.paths.sft_data
        )
        
        # Stage 7: Vision QA Generation
        self.add_stage(
            name="generate_vision_qa",
            description="Generate vision-based QA pairs",
            function=self._run_vision_qa_generation,
            dependencies=["extract_figures", "extract_abstracts"],
            output_path=self.config.paths.vlm_sft_data
        )
        
        # Build execution order based on dependencies
        self._build_execution_order()
    
    def add_stage(self, name: str, description: str, function: Optional[Callable] = None,
                  dependencies: List[str] = None, output_path: Optional[str] = None):
        """Add a stage to the pipeline"""
        stage = PipelineStage(
            name=name,
            description=description,
            function=function,
            dependencies=dependencies or [],
            output_path=output_path
        )
        self.stages[name] = stage
    
    def _build_execution_order(self):
        """Build the execution order based on dependencies using topological sort"""
        visited = set()
        order = []
        
        def visit(stage_name: str):
            if stage_name in visited:
                return
            
            stage = self.stages.get(stage_name)
            if not stage:
                return
            
            # Visit dependencies first
            for dep in stage.dependencies:
                if dep not in visited:
                    visit(dep)
            
            visited.add(stage_name)
            order.append(stage_name)
        
        # Visit all stages
        for stage_name in self.stages:
            visit(stage_name)
        
        self.execution_order = order
    
    def _check_stage_outputs(self, stage_name: str) -> bool:
        """Check if stage outputs already exist"""
        stage = self.stages[stage_name]
        
        if not stage.output_path:
            return False
        
        output_path = Path(stage.output_path)
        if not output_path.exists():
            return False
        
        # Check if directory has content
        if output_path.is_dir():
            files = list(output_path.iterdir())
            return len(files) > 0
        
        return output_path.is_file()
    
    def _can_run_stage(self, stage_name: str) -> bool:
        """Check if a stage can be run based on its dependencies"""
        stage = self.stages[stage_name]
        
        for dep in stage.dependencies:
            dep_stage = self.stages.get(dep)
            if not dep_stage or dep_stage.status != StageStatus.COMPLETED:
                return False
        
        return True
    
    def run(self, stages: Optional[List[str]] = None, skip_existing: bool = True,
            force: bool = False) -> PipelineResult:
        """
        Run the pipeline
        
        Args:
            stages: Specific stages to run (None = all stages)
            skip_existing: Skip stages if outputs already exist
            force: Force run even if outputs exist
        """
        start_time = time.time()
        
        # Determine which stages to run
        stages_to_run = stages or self.execution_order
        
        self.logger.info(f"Starting pipeline with stages: {', '.join(stages_to_run)}")
        
        # Reset stage statuses
        for stage_name in stages_to_run:
            self.stages[stage_name].status = StageStatus.PENDING
        
        # Run stages in order
        for stage_name in self.execution_order:
            if stage_name not in stages_to_run:
                continue
            
            stage = self.stages[stage_name]
            
            # Check if stage can be skipped
            if skip_existing and not force and self._check_stage_outputs(stage_name):
                self.logger.info(f"Skipping {stage_name}: outputs already exist")
                stage.status = StageStatus.SKIPPED
                continue
            
            # Check dependencies
            if not self._can_run_stage(stage_name):
                self.logger.warning(f"Cannot run {stage_name}: dependencies not met")
                stage.status = StageStatus.FAILED
                stage.error = "Dependencies not met"
                continue
            
            # Run the stage
            self.logger.info(f"Running stage: {stage_name} - {stage.description}")
            stage.status = StageStatus.RUNNING
            stage.start_time = datetime.now()
            
            try:
                if stage.function:
                    stage.function()
                    stage.status = StageStatus.COMPLETED
                    self.logger.info(f"Completed stage: {stage_name}")
                else:
                    self.logger.warning(f"No function defined for stage: {stage_name}")
                    stage.status = StageStatus.SKIPPED
            
            except Exception as e:
                self.logger.error(f"Failed stage {stage_name}: {str(e)}")
                stage.status = StageStatus.FAILED
                stage.error = str(e)
                
                if not force:
                    break  # Stop on failure unless forced
            
            finally:
                stage.end_time = datetime.now()
        
        # Calculate results
        total_time = time.time() - start_time
        
        result = PipelineResult(
            total_stages=len(stages_to_run),
            completed_stages=sum(1 for s in self.stages.values() 
                               if s.status == StageStatus.COMPLETED),
            failed_stages=sum(1 for s in self.stages.values() 
                            if s.status == StageStatus.FAILED),
            skipped_stages=sum(1 for s in self.stages.values() 
                             if s.status == StageStatus.SKIPPED),
            total_time=total_time,
            stage_results=self._get_stage_results()
        )
        
        self.logger.info(f"Pipeline completed in {total_time:.2f} seconds")
        self.logger.info(f"Results: {result.completed_stages} completed, "
                        f"{result.failed_stages} failed, {result.skipped_stages} skipped")
        
        return result
    
    def _get_stage_results(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed results for each stage"""
        results = {}
        
        for name, stage in self.stages.items():
            results[name] = {
                'status': stage.status.value,
                'description': stage.description,
                'start_time': stage.start_time.isoformat() if stage.start_time else None,
                'end_time': stage.end_time.isoformat() if stage.end_time else None,
                'duration': (stage.end_time - stage.start_time).total_seconds() 
                          if stage.start_time and stage.end_time else None,
                'error': stage.error,
                'output_path': stage.output_path
            }
        
        return results
    
    # Stage execution functions
    def _run_scraping(self):
        """Run the scraping stage"""
        from src.scrapers.nature_scraper import NatureScraper
        
        scraper = NatureScraper(output_dir=self.config.paths.nature_data)
        scraper.run()
    
    def _run_download(self):
        """Run the download stage"""
        # This would import and run the PDF downloader
        # For now, we'll use a placeholder
        self.logger.info("Running PDF download...")
        # from src.downloaders.pdf_downloader import PDFDownloader
        # downloader = PDFDownloader(self.config)
        # downloader.run()
    
    def _run_ocr(self):
        """Run the OCR stage"""
        self.logger.info("Running OCR processing...")
        
        # Run the chunked OCR script
        result = subprocess.run([
            sys.executable,
            'src/chunked-ocr.py'
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"OCR failed: {result.stderr}")
    
    def _run_figure_extraction(self):
        """Run the figure extraction stage"""
        self.logger.info("Running figure extraction...")
        # from src.processors.figure_extractor import FigureExtractor
        # extractor = FigureExtractor(self.config)
        # extractor.run()
    
    def _run_abstract_extraction(self):
        """Run the abstract extraction stage"""
        self.logger.info("Running abstract extraction...")
        # This would extract abstracts from OCR results
    
    def _run_text_qa_generation(self):
        """Run the text QA generation stage"""
        self.logger.info("Running text QA generation...")
        # from src.generators.text_qa_generator import TextQAGenerator
        # generator = TextQAGenerator(self.config)
        # generator.run()
    
    def _run_vision_qa_generation(self):
        """Run the vision QA generation stage"""
        self.logger.info("Running vision QA generation...")
        # from src.generators.vision_qa_generator import VisionQAGenerator
        # generator = VisionQAGenerator(self.config)
        # generator.run()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'stages': {
                name: {
                    'status': stage.status.value,
                    'description': stage.description,
                    'dependencies': stage.dependencies,
                    'output_exists': self._check_stage_outputs(name)
                }
                for name, stage in self.stages.items()
            },
            'execution_order': self.execution_order
        }
    
    def save_state(self, filepath: str):
        """Save pipeline state to file"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'project': self.config.project.__dict__,
                'paths': self.config.paths.__dict__
            },
            'stages': self._get_stage_results()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"Pipeline state saved to {filepath}")


def main():
    """Test the pipeline manager"""
    from src.config_manager import config
    
    config.load()
    pipeline = PipelineManager(config)
    
    # Get pipeline status
    status = pipeline.get_status()
    print("Pipeline Status:")
    for stage_name, stage_info in status['stages'].items():
        print(f"  {stage_name}: {stage_info['status']} - {stage_info['description']}")
    
    # Run specific stages
    # result = pipeline.run(stages=['scrape'], force=True)
    # print(f"\nPipeline Result: {result.completed_stages} completed, {result.failed_stages} failed")


if __name__ == '__main__':
    main()