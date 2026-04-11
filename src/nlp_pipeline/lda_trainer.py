import os
import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from src.nlp_pipeline.preprocessor import TextPreprocessor

class LDATrainer:
    def __init__(self, n_topics=10, max_features=5000):
        self.n_topics = n_topics
        self.max_features = max_features
        self.vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=max_features)
        self.lda = LatentDirichletAllocation(
            n_components=n_topics, 
            max_iter=10, 
            learning_method='online', 
            random_state=42
        )
        self.preprocessor = TextPreprocessor()

    def train(self, documents):
        """Trains the LDA model on a list of raw text documents."""
        print(f"Preprocessing {len(documents)} documents...")
        cleaned_docs = [self.preprocessor.preprocess(doc) for doc in tqdm(documents)]
        
        # Remove empty strings
        cleaned_docs = [doc for doc in cleaned_docs if doc.strip()]
        
        print("Vectorizing...")
        tf = self.vectorizer.fit_transform(cleaned_docs)
        
        print(f"Training LDA with {self.n_topics} topics...")
        self.lda.fit(tf)
        
        # Log top words for each topic
        self.log_topics()

    def transform(self, documents):
        """Infers topic distributions for new documents."""
        if not isinstance(documents, list):
            documents = [documents]
        cleaned_docs = [self.preprocessor.preprocess(doc) for doc in documents]
        tf = self.vectorizer.transform(cleaned_docs)
        return self.lda.transform(tf)

    def log_topics(self):
        feature_names = self.vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(self.lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
            print(f"Topic {topic_idx}: {', '.join(top_words)}")

    def save_model(self, path="models"):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(os.path.join(path, "lda_model.pkl"), "wb") as f:
            pickle.dump(self.lda, f)

    def load_model(self, path="models"):
        with open(os.path.join(path, "vectorizer.pkl"), "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(os.path.join(path, "lda_model.pkl"), "rb") as f:
            self.lda = pickle.load(f)

if __name__ == "__main__":
    # Test with some synthetic data
    docs = [
        "Focus on STEM, robotics, and coding for future engineers.",
        "A strong emphasis on performing arts, dance, and theater.",
        "Competitive athletics program with football and basketball teams.",
        "Academic excellence with AP courses and college preparation.",
        "Cultural immersion and language studies in Spanish and French."
    ] * 10
    
    trainer = LDATrainer(n_topics=5)
    trainer.train(docs)
    
    # Infer
    dist = trainer.transform("We want a school with coding and robotics.")
    print(f"Topic Distribution: {dist}")
