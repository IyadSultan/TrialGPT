import os
import sys
import traceback

# Force our custom NLTK data directory
nltk_data_dir = os.path.abspath('./nltk_data')
os.environ['NLTK_DATA'] = nltk_data_dir
print(f"NLTK_DATA set to: {nltk_data_dir}")

# Add tracing for imports
def trace_import(name, globals=None, locals=None, fromlist=(), level=0):
    if 'nltk' in name:
        print(f"Importing {name}")
        # Get the caller's frame info
        frame = sys._getframe(1)
        print(f"  called from {frame.f_code.co_filename}:{frame.f_lineno}")
    return original_import(name, globals, locals, fromlist, level)

# Save the original import
original_import = __builtins__.__import__
# Replace with our tracing version
__builtins__.__import__ = trace_import

try:
    import nltk
    print("NLTK search paths:", nltk.data.path)
    
    # Try loading some specific modules to see which one needs punkt_tab
    print("\nLoading different NLTK modules to identify the issue:")
    for module_name in ['nltk.tokenize', 'nltk.tokenize.punkt', 'nltk.corpus']:
        try:
            print(f"Importing {module_name}...")
            __import__(module_name)
            print(f"  {module_name} loaded successfully")
        except Exception as e:
            print(f"  Error loading {module_name}: {e}")
except Exception as e:
    print(f"Error importing NLTK: {e}") 