"""
Advanced Text Preprocessing Module
Includes NLP enhancements for better feature extraction
"""

import re
import pandas as pd
from typing import List, Dict
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class AdvancedPreprocessor:
    """
    Advanced text preprocessing with multiple cleaning strategies
    """
    
    def __init__(self, strategy='aggressive'):
        """
        Initialize preprocessor
        strategy: 'light', 'moderate', 'aggressive'
        """
        self.strategy = strategy
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """
        Main cleaning function based on strategy
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        if self.strategy in ['moderate', 'aggressive']:
            # Remove special characters
            text = re.sub(r'[^a-zA-Z\s\d]', ' ', text)
        
        if self.strategy == 'aggressive':
            # Remove numbers
            text = re.sub(r'\d+', '', text)
            
            # Remove stopwords
            tokens = word_tokenize(text)
            text = ' '.join([word for word in tokens if word not in self.stop_words])
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def expand_contractions(self, text: str) -> str:
        """
        Expand common contractions (e.g., "don't" -> "do not")
        """
        contractions_dict = {
            "ain't": "am not",
            "aren't": "are not",
            "can't": "cannot",
            "can't've": "cannot have",
            "could've": "could have",
            "couldn't": "could not",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'll": "he will",
            "he's": "he is",
            "how'd": "how did",
            "how'll": "how will",
            "how's": "how is",
            "i'd": "i would",
            "i'll": "i will",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it'd": "it would",
            "it'll": "it will",
            "it's": "it is",
            "shouldn't": "should not",
            "that's": "that is",
            "they'd": "they would",
            "they'll": "they will",
            "they're": "they are",
            "they've": "they have",
            "wasn't": "was not",
            "we'd": "we would",
            "we'll": "we will",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what's": "what is",
            "won't": "will not",
            "wouldn't": "would not",
            "you'd": "you would",
            "you'll": "you will",
            "you're": "you are",
            "you've": "you have"
        }
        
        pattern = re.compile(r'\b(' + '|'.join(contractions_dict.keys()) + r')\b')
        return pattern.sub(lambda x: contractions_dict[x.group()], text.lower())
    
    def remove_duplicate_chars(self, text: str) -> str:
        """
        Remove duplicate characters (e.g., "heeello" -> "helo")
        """
        return re.sub(r'(.)\1{2,}', r'\1', text)
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace
        """
        return ' '.join(text.split())
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract top keywords from text
        """
        tokens = word_tokenize(text.lower())
        # Remove stopwords
        keywords = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        # Count frequency
        from collections import Counter
        freq_dist = Counter(keywords)
        return [word for word, freq in freq_dist.most_common(top_n)]
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
        """
        Preprocess entire dataframe
        """
        df_processed = df.copy()
        
        for col in text_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna('').apply(
                    lambda x: self.clean_text(x)
                )
        
        return df_processed


def prepare_training_data(csv_path: str, strategy: str = 'moderate') -> tuple:
    """
    Load and prepare training data
    Returns: (X_text, y_labels)
    """
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Select columns
    if 'fraudulent' in df.columns:
        label_col = 'fraudulent'
    else:
        # Try to find label column
        label_col = [col for col in df.columns if 'fraud' in col.lower() or 'label' in col.lower()][0]
    
    # Initialize preprocessor
    preprocessor = AdvancedPreprocessor(strategy=strategy)
    
    # Clean text columns
    text_columns = ['title', 'description', 'company', 'location']
    text_columns = [col for col in text_columns if col in df.columns]
    
    # Preprocess
    df = preprocessor.preprocess_dataframe(df, text_columns)
    
    # Combine text
    df['combined_text'] = df[text_columns].fillna('').agg(' '.join, axis=1)
    
    X = df['combined_text']
    y = df[label_col]
    
    return X, y


if __name__ == "__main__":
    preprocessor = AdvancedPreprocessor(strategy='moderate')
    
    sample_text = """
    Don't miss this AMAZING opportunity!!! 
    Work from home with HUGE salaries!!!
    Visit https://scam-site.com for more info
    Contact us at fake@gmail.com
    """
    
    cleaned = preprocessor.clean_text(sample_text)
    print("Original:", sample_text)
    print("Cleaned:", cleaned)
    print("Keywords:", preprocessor.extract_keywords(cleaned))
