from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typedefs import Element


class VectorStore:
    def __init__(self):
        self.store: list[Element] = []

    def add(self, element: Element):
        self.store.append(element)

    def get(self, id: str) -> Element | None:
        return next((e for e in self.store if e["id"] == id), None)

    def query(self, q: list[float], k: int) -> list[Element]:
        if not self.store:
            return []

        # Convert query and embeddings to numpy arrays
        query_vec = np.array(q).reshape(1, -1)
        embeddings = np.array([e["embedding"] for e in self.store])

        # Compute cosine similarity between query and all embeddings
        similarities = cosine_similarity(query_vec, embeddings)[
            0]  # shape: (n_elements,)

        # Get indices of top-k most similar elements
        top_k_indices = similarities.argsort()[-k:][::-1]

        # Return top-k elements with metadata
        return [self.store[i] for i in top_k_indices]


