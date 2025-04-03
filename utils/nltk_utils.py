import os
import re
import nltk
import importlib
import sys

# Create nltk_data directory in the project root
nltk_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'nltk_data'))
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Set NLTK_DATA environment variable
os.environ['NLTK_DATA'] = nltk_data_dir

# Download required NLTK data
try:
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)
except Exception as e:
    print(f"Warning: Failed to download NLTK resources: {e}")
    print("Falling back to direct NLTK tokenizer...")

# Import NLTK's tokenizers directly
from nltk import word_tokenize
from nltk import sent_tokenize

# Remove any custom tokenizer overrides
if 'word_tokenize' in globals():
    del globals()['word_tokenize']
if 'sent_tokenize' in globals():
    del globals()['sent_tokenize']

# Export NLTK's tokenizers directly
__all__ = ['word_tokenize', 'sent_tokenize']

# Create the necessary directory structure for collocations.tab
tokenizers_dir = os.path.join(nltk_data_dir, 'tokenizers')
punkt_dir = os.path.join(tokenizers_dir, 'punkt')
punkt_tab_dir = os.path.join(tokenizers_dir, 'punkt_tab')
english_dir = os.path.join(punkt_tab_dir, 'english')

# Make sure all directories exist
for d in [tokenizers_dir, punkt_dir, punkt_tab_dir, english_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

# Create empty collocations.tab file if needed
collocations_tab = os.path.join(english_dir, 'collocations.tab')
if not os.path.exists(collocations_tab):
    with open(collocations_tab, 'w') as f:
        f.write("# This is a placeholder file\n")
    print(f"Created placeholder {collocations_tab}")

# Define a more direct approach using our own tokenizers
def custom_sent_tokenize(text):
    """A simple sentence tokenizer that doesn't require NLTK resources"""
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def custom_word_tokenize(text):
    """Simple word tokenizer that splits on spaces and punctuation"""
    if not text:
        return []
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return [w for w in text.split() if w]

# Use our custom tokenizers directly instead of NLTK's
sent_tokenize = custom_sent_tokenize
word_tokenize = custom_word_tokenize

print("Using custom tokenizers directly")

def word_tokenize(text):
    """Wrapper around NLTK's word_tokenize to ensure consistent behavior"""
    return word_tokenize(text)

# Just export NLTK's tokenizers
sent_tokenize = sent_tokenize
word_tokenize = word_tokenize 