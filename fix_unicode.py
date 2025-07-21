"""
Fix Unicode encoding issues by replacing emoji characters with plain text
"""
import os
import re

def fix_emojis_in_file(filepath):
    """Replace emoji characters with plain text equivalents"""
    
    # Read the file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define emoji replacements
    replacements = {
        '🧪': '[TEST]',
        '✅': '[PASS]',
        '❌': '[FAIL]', 
        '🎉': '[SUCCESS]',
        '💡': '[INFO]',
        '📝': '[NOTE]',
        '📋': '[LIST]',
        '🤖': '[AI]',
        '🔧': '[CONFIG]',
        '🔄': '[PROCESSING]',
        '🔍': '[SEARCH]',
        '🚀': '[START]'
    }
    
    # Replace emojis
    for emoji, replacement in replacements.items():
        content = content.replace(emoji, replacement)
    
    # Write back the file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed emojis in: {filepath}")

def main():
    test_files = [
        'tests/test_simple.py',
        'tests/test_llm.py',
        'tests/run_tests.py'
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            fix_emojis_in_file(file_path)
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    main()
