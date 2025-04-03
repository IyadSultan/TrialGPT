import os
import sys
import traceback

# Monkey patch Python's import system to catch imports
original_import = __import__

def debug_import(*args, **kwargs):
    module_name = args[0]
    if 'nltk' in module_name:
        frame = traceback.extract_stack()[-2]
        print(f"NLTK import detected in: {frame.filename}:{frame.lineno}")
        print(f"Importing: {module_name}")
    return original_import(*args, **kwargs)

sys.meta_path.insert(0, debug_import)

# Print current NLTK_DATA environment variable
print(f"NLTK_DATA is set to: {os.environ.get('NLTK_DATA', 'Not set')}") 