import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
def download_nltk_resources():
    resources = ['stopwords', 'punkt', 'wordnet', 'punkt_tab']
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except Exception as e:
            print(f"Warning: Failed to download {res}: {e}")

download_nltk_resources()

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        # Add some domain-specific stopwords
        self.stop_words.update(['school', 'education', 'student', 'students', 'learning', 'academic', 'program', 'programs'])

    def preprocess(self, text):
        if not text or not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove non-alphabetic characters
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words, then lemmatize
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return " ".join(cleaned_tokens)

if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    sample_text = "The school offers a great STEM program and focuses on athletics and arts."
    print(f"Original: {sample_text}")
    print(f"Cleaned: {preprocessor.preprocess(sample_text)}")
