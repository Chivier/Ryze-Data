#!/usr/bin/env python3
"""
Test runner script for Ryze-Data
Provides convenient commands for running different test suites
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class TestRunner:
    """Manages test execution with different configurations"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_dir = self.project_root / 'tests'
    
    def run_pytest(self, args: List[str], env: Optional[dict] = None) -> int:
        """Run pytest with given arguments"""
        cmd = [sys.executable, '-m', 'pytest'] + args
        
        # Set up environment
        test_env = os.environ.copy()
        test_env['PYTHONPATH'] = str(self.project_root)
        test_env['TESTING'] = 'true'
        
        if env:
            test_env.update(env)
        
        # Run tests
        print(f"Running: {' '.join(cmd)}")
        print("-" * 60)
        
        result = subprocess.run(cmd, env=test_env)
        return result.returncode
    
    def run_unit_tests(self, verbose: bool = False) -> int:
        """Run unit tests only"""
        print("\n🧪 Running Unit Tests...")
        args = [
            'tests/unit',
            '-m', 'unit',
            '--cov=src',
            '--cov-report=term-missing'
        ]
        
        if verbose:
            args.append('-vv')
        
        return self.run_pytest(args)
    
    def run_integration_tests(self, verbose: bool = False) -> int:
        """Run integration tests only"""
        print("\n🔗 Running Integration Tests...")
        args = [
            'tests/integration',
            '-m', 'integration'
        ]
        
        if verbose:
            args.append('-vv')
        
        return self.run_pytest(args)
    
    def run_sample_tests(self, verbose: bool = False) -> int:
        """Run tests with sample data"""
        print("\n📊 Running Sample Data Tests...")
        
        # Copy sample data if exists
        self._setup_sample_data()
        
        args = [
            'tests',
            '-m', 'sample',
            '-k', 'sample'
        ]
        
        if verbose:
            args.append('-vv')
        
        return self.run_pytest(args)
    
    def run_quick_tests(self) -> int:
        """Run quick tests (no slow tests, no API calls)"""
        print("\n⚡ Running Quick Tests...")
        args = [
            'tests',
            '-m', 'not slow and not requires_api',
            '--tb=short',
            '-q'
        ]
        
        return self.run_pytest(args)
    
    def run_all_tests(self, verbose: bool = False) -> int:
        """Run all tests with coverage"""
        print("\n🎯 Running All Tests...")
        args = [
            'tests',
            '--cov=src',
            '--cov-report=html',
            '--cov-report=term-missing'
        ]
        
        if verbose:
            args.append('-vv')
        
        return self.run_pytest(args)
    
    def run_specific_test(self, test_path: str, verbose: bool = False) -> int:
        """Run a specific test file or test case"""
        print(f"\n🎯 Running Specific Test: {test_path}")
        args = [test_path]
        
        if verbose:
            args.append('-vv')
        
        return self.run_pytest(args)
    
    def run_pipeline_test(self) -> int:
        """Run pipeline test with single article"""
        print("\n🚀 Running Pipeline Test with Sample Article...")
        
        # Set up test environment
        env = {
            'RYZE_MAX_PAPERS': '1',
            'RYZE_QA_RATIO': '2',
            'RYZE_NUM_WORKERS': '1'
        }
        
        args = [
            'tests/integration/test_full_pipeline.py::TestEndToEndScenarios::test_process_biocot_sample_data',
            '-vv',
            '--tb=short'
        ]
        
        return self.run_pytest(args, env)
    
    def _setup_sample_data(self):
        """Set up sample data for testing"""
        # Check if biocot sample data exists
        biocot_sample = self.project_root / 'biocot' / 'data-sample'
        test_data = self.test_dir / 'data'
        
        if biocot_sample.exists() and not test_data.exists():
            print(f"Setting up sample data from {biocot_sample}...")
            
            # Create test data directories
            test_data.mkdir(parents=True, exist_ok=True)
            
            # Copy sample files
            import shutil
            
            # Copy OCR results
            ocr_source = biocot_sample / 'ocr_result'
            if ocr_source.exists():
                ocr_dest = test_data / 'ocr_results'
                if not ocr_dest.exists():
                    shutil.copytree(ocr_source, ocr_dest)
            
            # Copy other sample data
            for item in ['abstract', 'sft_data']:
                source = biocot_sample / item
                if source.exists():
                    dest = test_data / item
                    if not dest.exists():
                        shutil.copytree(source, dest)
    
    def clean_test_data(self):
        """Clean up test data and artifacts"""
        print("\n🧹 Cleaning test data...")
        
        # Remove test data directory
        test_data = self.test_dir / 'data'
        if test_data.exists():
            import shutil
            shutil.rmtree(test_data)
            print(f"Removed {test_data}")
        
        # Remove test logs
        test_logs = self.test_dir / 'logs'
        if test_logs.exists():
            import shutil
            shutil.rmtree(test_logs)
            print(f"Removed {test_logs}")
        
        # Remove coverage reports
        htmlcov = self.project_root / 'htmlcov'
        if htmlcov.exists():
            import shutil
            shutil.rmtree(htmlcov)
            print(f"Removed {htmlcov}")
        
        # Remove pytest cache
        pytest_cache = self.project_root / '.pytest_cache'
        if pytest_cache.exists():
            import shutil
            shutil.rmtree(pytest_cache)
            print(f"Removed {pytest_cache}")
        
        print("Test data cleaned!")


def main():
    """Main entry point for test runner"""
    parser = argparse.ArgumentParser(description='Ryze-Data Test Runner')
    
    subparsers = parser.add_subparsers(dest='command', help='Test commands')
    
    # Unit tests
    unit_parser = subparsers.add_parser('unit', help='Run unit tests')
    unit_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    # Integration tests
    integration_parser = subparsers.add_parser('integration', help='Run integration tests')
    integration_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    # Sample tests
    sample_parser = subparsers.add_parser('sample', help='Run tests with sample data')
    sample_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    # Quick tests
    subparsers.add_parser('quick', help='Run quick tests (no slow/API tests)')
    
    # All tests
    all_parser = subparsers.add_parser('all', help='Run all tests')
    all_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    # Specific test
    specific_parser = subparsers.add_parser('specific', help='Run specific test')
    specific_parser.add_argument('test_path', help='Path to test file or test case')
    specific_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    # Pipeline test
    subparsers.add_parser('pipeline', help='Run pipeline test with sample article')
    
    # Clean command
    subparsers.add_parser('clean', help='Clean test data and artifacts')
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if not args.command:
        # Default to quick tests
        return runner.run_quick_tests()
    
    if args.command == 'unit':
        return runner.run_unit_tests(args.verbose)
    
    elif args.command == 'integration':
        return runner.run_integration_tests(args.verbose)
    
    elif args.command == 'sample':
        return runner.run_sample_tests(args.verbose)
    
    elif args.command == 'quick':
        return runner.run_quick_tests()
    
    elif args.command == 'all':
        return runner.run_all_tests(args.verbose)
    
    elif args.command == 'specific':
        return runner.run_specific_test(args.test_path, args.verbose)
    
    elif args.command == 'pipeline':
        return runner.run_pipeline_test()
    
    elif args.command == 'clean':
        runner.clean_test_data()
        return 0
    
    return 0


if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════╗
║        Ryze-Data Test Runner             ║
╚══════════════════════════════════════════╝
    """)
    
    sys.exit(main())