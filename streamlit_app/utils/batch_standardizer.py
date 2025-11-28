"""
Batch Page Standardization Script - Task 7

This script applies quick standardization fixes to all remaining pages:
- Adds traceback import if missing
- Ensures consistent import structure
- Reports any pages that need manual attention
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any

class BatchPageStandardizer:
    """Applies batch standardization to page modules."""
    
    def __init__(self, pages_dir: str):
        self.pages_dir = Path(pages_dir)
        self.results = {}
        self.target_pages = [
            'analytics.py',
            'data_training.py', 
            'history.py',
            'incremental_learning.py',
            'prediction_ai.py',
            'prediction_engine.py',
            'settings.py',
            'model_manager.py'
        ]
    
    def standardize_all_pages(self) -> Dict[str, Any]:
        """Apply batch standardization to all target pages."""
        print("Starting batch standardization...")
        
        for page_file in self.target_pages:
            page_path = self.pages_dir / page_file
            if page_path.exists():
                print(f"Processing {page_file}...")
                self.results[page_file] = self._standardize_page(page_path)
            else:
                self.results[page_file] = {'status': 'not_found', 'message': 'File not found'}
        
        return self._generate_report()
    
    def _standardize_page(self, page_path: Path) -> Dict[str, Any]:
        """Standardize a single page file."""
        try:
            # Read the file
            with open(page_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            changes_made = []
            
            # Check if traceback import is missing
            if 'import traceback' not in content and 'from traceback import' not in content:
                # Find where to add the traceback import
                lines = content.split('\n')
                import_section_end = 0
                
                # Find the end of the import section
                for i, line in enumerate(lines):
                    if line.strip().startswith(('import ', 'from ')) and not line.strip().startswith('from ..'):
                        import_section_end = i + 1
                    elif line.strip() == '' and import_section_end > 0:
                        break
                    elif not line.strip().startswith(('import ', 'from ', '#', '"""')) and line.strip() != '' and import_section_end > 0:
                        break
                
                # Insert traceback import
                if import_section_end > 0:
                    lines.insert(import_section_end, 'import traceback')
                    content = '\n'.join(lines)
                    changes_made.append('Added traceback import')
                else:
                    # If we can't find a good place, add it after the docstring
                    docstring_end = content.find('"""', content.find('"""') + 3)
                    if docstring_end > 0:
                        docstring_end += 3
                        content = content[:docstring_end] + '\n\nimport traceback' + content[docstring_end:]
                        changes_made.append('Added traceback import after docstring')
            
            # Write back if changes were made
            if changes_made:
                with open(page_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return {
                    'status': 'modified',
                    'changes': changes_made,
                    'message': f'Applied {len(changes_made)} changes'
                }
            else:
                return {
                    'status': 'no_changes',
                    'message': 'No changes needed'
                }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error processing file: {str(e)}'
            }
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate standardization report."""
        total_pages = len(self.results)
        modified_pages = sum(1 for r in self.results.values() if r['status'] == 'modified')
        error_pages = sum(1 for r in self.results.values() if r['status'] == 'error')
        
        return {
            'summary': {
                'total_pages': total_pages,
                'modified_pages': modified_pages,
                'error_pages': error_pages,
                'success_rate': f"{((total_pages - error_pages) / total_pages * 100):.1f}%" if total_pages > 0 else "0%"
            },
            'details': self.results
        }


def main():
    """Main standardization function."""
    pages_dir = r"c:\Users\dian_\OneDrive\1 - My Documents\9 - Rocket Innovations Inc\gaming-ai-bot\streamlit_app\pages"
    
    standardizer = BatchPageStandardizer(pages_dir)
    report = standardizer.standardize_all_pages()
    
    print("\n" + "="*60)
    print("BATCH STANDARDIZATION REPORT")
    print("="*60)
    
    # Summary
    summary = report['summary']
    print(f"\nSUMMARY:")
    print(f"   • Total pages processed: {summary['total_pages']}")
    print(f"   • Pages modified: {summary['modified_pages']}")
    print(f"   • Pages with errors: {summary['error_pages']}")
    print(f"   • Success rate: {summary['success_rate']}")
    
    # Details
    print(f"\nDETAILS:")
    for page_name, result in report['details'].items():
        status_icon = "✅" if result['status'] in ['modified', 'no_changes'] else "❌"
        print(f"   {status_icon} {page_name}: {result['message']}")
        if result.get('changes'):
            for change in result['changes']:
                print(f"      - {change}")
    
    return report

if __name__ == "__main__":
    main()