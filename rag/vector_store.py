import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class FAISSRetrieval:
    def __init__(self, documents):
        self.documents = documents
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.embeddings = self.model.encode(documents)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings).astype("float32"))

    def retrieve(self, query, top_k=5):
        q = self.model.encode([query]).astype("float32")
        _, idx = self.index.search(q, top_k)

        docs = [self.documents[i] for i in idx[0]]
        return docs