#!/usr/bin/env python3
"""
Batch fix script for page import issues

This script fixes the app_log import issue in all affected pages by:
1. Moving the fallback app_log definition before the try-except block
2. Ensuring proper error handling
"""

import os
from pathlib import Path

def fix_app_log_issue(file_path: Path):
    """Fix app_log import issue in a single file"""
    
    print(f"🔧 Fixing {file_path.name}...")
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if this file has the problematic pattern
        if 'from streamlit_app.core.app_logging import app_log' not in content:
            print(f"  ✅ No app_log import found - skipping")
            return True
            
        if 'class app_log:' not in content:
            print(f"  ✅ No fallback app_log class found - skipping")
            return True
        
        # Find the try block
        try_start = content.find('try:\n    # Import enhanced infrastructure components')
        if try_start == -1:
            try_start = content.find('try:\n    from streamlit_app.core')
        
        if try_start == -1:
            print(f"  ⚠️ No matching try block pattern found")
            return False
            
        # Find the except block
        except_start = content.find('except ImportError as e:', try_start)
        if except_start == -1:
            print(f"  ⚠️ No matching except block found")
            return False
        
        # Find the fallback app_log class
        fallback_start = content.find('    class app_log:', except_start)
        if fallback_start == -1:
            print(f"  ⚠️ No fallback app_log class found in except block")
            return False
            
        # Find the end of the fallback class
        fallback_end = content.find('\n\n', fallback_start)
        if fallback_end == -1:
            # Look for the end differently
            lines = content[fallback_start:].split('\n')
            class_lines = []
            for i, line in enumerate(lines):
                if i == 0 or line.startswith('        ') or line.strip() == '':
                    class_lines.append(line)
                else:
                    break
            fallback_class = '\n'.join(class_lines)
            fallback_end = fallback_start + len(fallback_class)
        else:
            fallback_class = content[fallback_start:fallback_end]
        
        # Extract the fallback class definition
        fallback_class = content[fallback_start:fallback_end].strip()
        
        # Remove the fallback class from the except block
        new_except_block = content[:fallback_start] + content[fallback_end:]
        
        # Insert the fallback class before the try block
        fallback_def = f"# Define fallback logging first to avoid import issues\nclass app_log:\n    @staticmethod\n    def error(msg): print(f\"ERROR: {{msg}}\")\n    @staticmethod  \n    def warning(msg): print(f\"WARNING: {{msg}}\")\n    @staticmethod\n    def info(msg): print(f\"INFO: {{msg}}\")\n\n"
        
        # Insert before the try block
        new_content = new_except_block[:try_start] + fallback_def + new_except_block[try_start:]
        
        # Write the fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
        print(f"  ✅ Fixed successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Error fixing {file_path.name}: {e}")
        return False

def main():
    """Fix all pages with app_log import issues"""
    
    # Get the project root
    project_root = Path(__file__).parent
    pages_dir = project_root / "streamlit_app" / "pages"
    
    pages_to_fix = [
        'data_training.py',
        'help_docs.py', 
        'history.py',
        'incremental_learning.py',
        'model_manager.py',
        'prediction_ai.py',
        'prediction_engine.py',
        'settings.py'
    ]
    
    print("🔧 Batch fixing page import issues")
    print("=" * 50)
    
    results = []
    for page_name in pages_to_fix:
        page_path = pages_dir / page_name
        if page_path.exists():
            success = fix_app_log_issue(page_path)
            results.append(success)
        else:
            print(f"⚠️ {page_name} not found - skipping")
            results.append(False)
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {sum(results)}/{len(results)} pages fixed successfully")
    
    if all(results):
        print("🎉 All pages fixed! Ready for testing.")
    else:
        print("⚠️ Some pages couldn't be fixed automatically.")

if __name__ == "__main__":
    main()