"""
retriever.py -- TF-IDF cosine similarity retriever (no GPU required).
"""
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

WIKIPEDIA_2023 = {
    "cambodia": (
        "Cambodia held elections in 2023. Hun Sen transitioned power to Hun Manet. "
        "The Cambodian People's Party won majority seats."
    ),
    "guatemala": (
        "Guatemala held elections in 2023. Electoral fraud allegations arose. "
        "The opening of electoral boxes confirmed the outcome. "
        "The result was certified by electoral authorities."
    ),
    "thailand": (
        "Thailand held elections in 2023. Move Forward Party won the most seats. "
        "Srettha Thavisin became prime minister. "
        "The result was certified by the election commission."
    ),
    "turkey": (
        "Turkey held presidential elections in 2023. Recep Tayyip Erdogan won re-election. "
        "The result was confirmed by the Supreme Electoral Board."
    ),
    "spain": (
        "Spain held general elections in July 2023. PP won most seats. "
        "Pedro Sanchez remained prime minister after coalition talks. "
        "The result was certified by the Interior Ministry."
    ),
    "nigeria": (
        "Nigeria held presidential elections in 2023. Bola Tinubu was elected president. "
        "The result was upheld by Nigerian courts."
    ),
    "argentina": (
        "Argentina held elections in 2023. Javier Milei won the presidential runoff. "
        "The result was certified by Argentine electoral authorities."
    ),
    "taiwan": (
        "Taiwan held presidential elections in January 2024. Lai Ching-te was elected president. "
        "The KMT lost. The result was certified by the CEC."
    ),
    "poland": (
        "Poland held parliamentary elections in 2023. Donald Tusk's coalition won. "
        "PiS conceded. The result was certified by the electoral commission."
    ),
    "greece": (
        "Greece held parliamentary elections in June 2023. New Democracy won outright majority. "
        "Kyriakos Mitsotakis remained prime minister. "
        "The result was certified by the Interior Ministry."
    ),
}


class Retriever:
    def __init__(self, corpus):
        self.corpus = corpus
        self._vectorizer = None
        self._doc_vectors = None
        self._keys = []
        if corpus:
            self._build()

    def _build(self):
        self._keys = list(self.corpus.keys())
        texts = [self.corpus[k] for k in self._keys]
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._doc_vectors = self._vectorizer.fit_transform(texts)

    def retrieve(self, query, top_k=3):
        if self._vectorizer is None:
            return []
        q_vec = self._vectorizer.transform([query])
        scores = sk_cosine(q_vec, self._doc_vectors)[0]
        paired = list(zip(self._keys, scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        results = []
        for key, score in paired[:top_k]:
            if score > 0.0:
                results.append((key, self.corpus[key], float(score)))
        return results

    def retrieve_context(self, query, top_k=3):
        hits = self.retrieve(query, top_k=top_k)
        if not hits:
            return ""
        parts = [f"[{k}] {text}" for k, text, _ in hits if _ > 0.05]
        return "\n".join(parts)


def build_election_retriever():
    return Retriever(WIKIPEDIA_2023)


def build_empty_retriever():
    return Retriever({})
