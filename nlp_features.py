"""
NLP Feature Extraction Module
Extracts advanced NLP features for fake job detection
"""

import re
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

class NLPFeatureExtractor:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
    
    def extract_sentiment_features(self, text):
        """
        Extract sentiment analysis features
        Returns: dict with sentiment metrics
        """
        # TextBlob sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # VADER sentiment (better for social media/informal text)
        vader_scores = self.vader_analyzer.polarity_scores(text)
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_compound': vader_scores['compound']
        }
    
    def extract_linguistic_features(self, text):
        """
        Extract linguistic features that correlate with fraud
        """
        tokens = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        features = {
            'text_length': len(text),
            'num_tokens': len(tokens),
            'num_sentences': len(sentences),
            'avg_sentence_length': len(tokens) / len(sentences) if sentences else 0,
            'num_unique_words': len(set(tokens)),
            'lexical_diversity': len(set(tokens)) / len(tokens) if tokens else 0,
            'num_stopwords': sum(1 for token in tokens if token in self.stop_words),
            'num_punctuation': sum(1 for char in text if char in '!?.,;:'),
            'num_uppercase': sum(1 for char in text if char.isupper()),
            'num_digits': sum(1 for char in text if char.isdigit()),
            'num_special_chars': sum(1 for char in text if not char.isalnum() and char != ' ')
        }
        
        return features
    
    def extract_suspicious_patterns(self, text):
        """
        Extract patterns commonly found in fraudulent job postings
        """
        text_lower = text.lower()
        
        # Suspicious keywords/patterns
        urgent_keywords = ['urgent', 'immediately', 'asap', 'no experience needed', 'easy money']
        too_good_keywords = ['no work', 'work from home', 'easy', 'passive income', 'make money fast']
        contact_issues = ['email only', 'whatsapp', 'telegram', 'western union', 'money transfer']
        
        features = {
            'has_urgent_language': sum(1 for kw in urgent_keywords if kw in text_lower),
            'has_too_good_claims': sum(1 for kw in too_good_keywords if kw in text_lower),
            'has_suspicious_contact': sum(1 for kw in contact_issues if kw in text_lower),
            'has_generic_description': len(text) < 100,
            'has_multiple_exclamations': text.count('!') > 3,
            'has_excessive_caps': sum(1 for char in text if char.isupper()) > len(text) * 0.3,
            'has_url_presence': bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
            'has_email_presence': bool(re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text))
        }
        
        return features
    
    def extract_named_entities_features(self, text):
        """
        Extract features based on named entity patterns
        """
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        
        features = {
            'num_proper_nouns': sum(1 for word, tag in pos_tags if tag in ['NNP', 'NNPS']),
            'num_verbs': sum(1 for word, tag in pos_tags if tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']),
            'num_nouns': sum(1 for word, tag in pos_tags if tag in ['NN', 'NNS']),
            'num_adjectives': sum(1 for word, tag in pos_tags if tag in ['JJ', 'JJR', 'JJS']),
        }
        
        return features
    
    def extract_salary_features(self, salary_text):
        """
        Extract features from salary information
        """
        if not salary_text or salary_text.strip() == '':
            return {
                'has_salary_info': False,
                'salary_mentioned': 0,
                'salary_too_high': False,
                'salary_range_present': False
            }
        
        salary_lower = salary_text.lower()
        salary_numbers = re.findall(r'\d+[\d,]*', salary_text)
        
        features = {
            'has_salary_info': len(salary_numbers) > 0,
            'salary_mentioned': len(salary_numbers),
            'salary_too_high': any(int(num.replace(',', '')) > 500000 for num in salary_numbers if num.replace(',', '').isdigit()),
            'salary_range_present': '-' in salary_text or 'to' in salary_lower
        }
        
        return features
    
    def extract_all_features(self, title='', company='', location='', salary='', description=''):
        """
        Extract all NLP features from job posting
        """
        full_text = f"{title} {company} {location} {description}"
        
        all_features = {}
        
        # Sentiment
        all_features.update({f'nlp_sentiment_{k}': v for k, v in 
                            self.extract_sentiment_features(full_text).items()})
        
        # Linguistic
        all_features.update({f'nlp_linguistic_{k}': v for k, v in 
                            self.extract_linguistic_features(full_text).items()})
        
        # Suspicious patterns
        all_features.update({f'nlp_suspicious_{k}': v for k, v in 
                            self.extract_suspicious_patterns(full_text).items()})
        
        # Named entities
        all_features.update({f'nlp_entities_{k}': v for k, v in 
                            self.extract_named_entities_features(full_text).items()})
        
        # Salary
        all_features.update({f'nlp_salary_{k}': v for k, v in 
                            self.extract_salary_features(salary).items()})
        
        return all_features


if __name__ == "__main__":
    # Test the feature extractor
    extractor = NLPFeatureExtractor()
    
    sample_text = {
        'title': 'Work from Home - Easy Money!',
        'company': 'XYZ Corp',
        'location': 'Remote',
        'salary': '$5000 - $10000',
        'description': 'URGENT! No experience needed. Make passive income easily! Email us ASAP!!!'
    }
    
    features = extractor.extract_all_features(**sample_text)
    
    print("Extracted Features:")
    for key, value in features.items():
        print(f"{key}: {value}")
