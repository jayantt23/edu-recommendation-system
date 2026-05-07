import os
import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from src.nlp_pipeline.preprocessor import TextPreprocessor

class LDATrainer:
    # UPGRADE: Added alpha (doc_topic_prior) and eta (topic_word_prior)
    def __init__(self, n_topics=5, max_features=5000, alpha=0.1, eta=0.01):
        self.n_topics = n_topics
        self.max_features = max_features
        
        # UPGRADE: Tightened max_df to 0.85 to aggressively remove "boilerplate" words 
        # (like "school", "students") that muddy the topics.
        self.vectorizer = CountVectorizer(max_df=0.85, min_df=2, max_features=max_features)
        
        # UPGRADE: Explicit Dirichlet Hyperparameters for Sparse Granularity
        self.lda = LatentDirichletAllocation(
            n_components=n_topics, 
            doc_topic_prior=alpha,  # Alpha: Forces schools to have 1-2 distinct identities
            topic_word_prior=eta,   # Eta: Forces words to strongly belong to specific topics
            max_iter=15,            # Slight bump for better convergence
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
        
        print(f"Training LDA with {self.n_topics} topics (Alpha={self.lda.doc_topic_prior}, Eta={self.lda.topic_word_prior})...")
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
    docs = [
        "Focus on STEM, robotics, and coding for future engineers.",
        "A strong emphasis on performing arts, dance, and theater.",
        "Competitive athletics program with football and basketball teams.",
        "Academic excellence with AP courses and college preparation.",
        "Cultural immersion and language studies in Spanish and French."
    ] * 10
    
    trainer = LDATrainer(n_topics=5)
    trainer.train(docs)
    
    dist = trainer.transform("We want a school with coding and robotics.")
    print(f"Topic Distribution: {np.round(dist, 3)}")