"""
📋 Page Structure Analyzer - Task 7 Standardization Tool

This script analyzes all pages in the streamlit_app/pages directory to identify
standardization requirements and creates a comprehensive standardization plan.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

class PageStructureAnalyzer:
    """Analyzes page structure for standardization requirements."""
    
    def __init__(self, pages_dir: str):
        self.pages_dir = Path(pages_dir)
        self.pages_analysis = {}
        self.standardization_issues = []
        
    def analyze_all_pages(self) -> Dict[str, Any]:
        """Analyze all pages and generate standardization report."""
        
        # Get all Python files in pages directory
        page_files = [f for f in self.pages_dir.glob("*.py") 
                     if not f.name.startswith("__") and f.name != "page_template.py"]
        
        print(f"Analyzing {len(page_files)} page files...")
        
        for page_file in page_files:
            self.pages_analysis[page_file.name] = self._analyze_page(page_file)
        
        return self._generate_report()
    
    def _analyze_page(self, page_file: Path) -> Dict[str, Any]:
        """Analyze a single page file."""
        try:
            with open(page_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            analysis = {
                'file_name': page_file.name,
                'file_path': str(page_file),
                'line_count': len(content.splitlines()),
                'render_functions': [],
                'import_patterns': [],
                'issues': []
            }
            
            # Check for render function patterns
            render_functions = re.findall(r'def (render_\w*)', content)
            analysis['render_functions'] = render_functions
            
            # Check for standard render_page function
            if 'render_page' not in render_functions:
                analysis['issues'].append("Missing standard render_page() function")
            
            # Check for old function patterns
            old_patterns = ['render_dashboard_page', 'render_predictions_page', 'render_help_documentation_page']
            for pattern in old_patterns:
                if pattern in content:
                    analysis['issues'].append(f"Uses old function name: {pattern}")
            
            # Check import patterns
            import_lines = [line.strip() for line in content.splitlines() 
                          if line.strip().startswith(('import ', 'from '))]
            analysis['import_patterns'] = import_lines[:10]  # First 10 imports
            
            # Check for standard imports
            standard_imports = ['streamlit as st', 'traceback', 'typing']
            for std_import in standard_imports:
                if not any(std_import in imp for imp in import_lines):
                    analysis['issues'].append(f"Missing standard import: {std_import}")
            
            # Check for error handling pattern
            if 'try:' not in content or 'except Exception' not in content:
                analysis['issues'].append("Missing standard error handling pattern")
            
            # Check for logging
            if 'app_log' not in content:
                analysis['issues'].append("Missing logging integration")
            
            # Check parameter patterns
            param_patterns = [
                'services_registry',
                'ai_engines', 
                'components'
            ]
            param_issues = []
            for param in param_patterns:
                if param not in content:
                    param_issues.append(param)
            
            if param_issues:
                analysis['issues'].append(f"Missing expected parameters: {', '.join(param_issues)}")
            
            return analysis
            
        except Exception as e:
            return {
                'file_name': page_file.name,
                'error': f"Failed to analyze: {str(e)}",
                'issues': [f"Analysis failed: {str(e)}"]
            }
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive standardization report."""
        
        total_pages = len(self.pages_analysis)
        pages_with_issues = sum(1 for analysis in self.pages_analysis.values() 
                               if analysis.get('issues', []))
        
        report = {
            'summary': {
                'total_pages': total_pages,
                'pages_with_issues': pages_with_issues,
                'standardization_progress': f"{((total_pages - pages_with_issues) / total_pages * 100):.1f}%" if total_pages > 0 else "0%"
            },
            'pages': self.pages_analysis,
            'prioritized_fixes': self._prioritize_fixes()
        }
        
        return report
    
    def _prioritize_fixes(self) -> List[Dict[str, Any]]:
        """Prioritize standardization fixes by importance and impact."""
        
        fixes = []
        
        for file_name, analysis in self.pages_analysis.items():
            if analysis.get('issues'):
                fixes.append({
                    'file': file_name,
                    'issues': analysis['issues'],
                    'priority': self._calculate_priority(analysis['issues']),
                    'line_count': analysis.get('line_count', 0)
                })
        
        # Sort by priority (higher priority first)
        fixes.sort(key=lambda x: x['priority'], reverse=True)
        
        return fixes
    
    def _calculate_priority(self, issues: List[str]) -> int:
        """Calculate priority score for fixes."""
        priority = 0
        
        for issue in issues:
            if "Missing standard render_page()" in issue:
                priority += 10  # Highest priority
            elif "old function name" in issue:
                priority += 8   # High priority
            elif "Missing standard error handling" in issue:
                priority += 7   # High priority
            elif "Missing standard import" in issue:
                priority += 5   # Medium priority
            elif "Missing expected parameters" in issue:
                priority += 6   # Medium-high priority
            elif "Missing logging" in issue:
                priority += 3   # Lower priority
        
        return priority


def main():
    """Main analysis function."""
    pages_dir = r"c:\Users\dian_\OneDrive\1 - My Documents\9 - Rocket Innovations Inc\gaming-ai-bot\streamlit_app\pages"
    
    analyzer = PageStructureAnalyzer(pages_dir)
    report = analyzer.analyze_all_pages()
    
    print("\n" + "="*80)
    print("PAGE STRUCTURE ANALYSIS REPORT")
    print("="*80)
    
    # Summary
    summary = report['summary']
    print(f"\nSUMMARY:")
    print(f"   • Total pages analyzed: {summary['total_pages']}")
    print(f"   • Pages needing fixes: {summary['pages_with_issues']}")
    print(f"   • Standardization progress: {summary['standardization_progress']}")
    
    # Prioritized fixes
    print(f"\nPRIORITIZED FIXES:")
    for i, fix in enumerate(report['prioritized_fixes'][:10], 1):  # Top 10
        print(f"\n   {i}. {fix['file']} (Priority: {fix['priority']}, Lines: {fix['line_count']})")
        for issue in fix['issues'][:3]:  # Top 3 issues per file
            print(f"      {issue}")
        if len(fix['issues']) > 3:
            print(f"      ... and {len(fix['issues']) - 3} more issues")
    
    # Detailed pages info
    print(f"\nDETAILED PAGES INFO:")
    for file_name, analysis in report['pages'].items():
        if analysis.get('issues'):
            print(f"\n   {file_name}:")
            print(f"      Lines: {analysis.get('line_count', 'Unknown')}")
            print(f"      Render functions: {', '.join(analysis.get('render_functions', []))}")
            print(f"      Issues: {len(analysis.get('issues', []))}")
    
    return report

if __name__ == "__main__":
    main()