class Answer:
    def __init__(self, answer, contexts):
        self.answer = answer
        self.contexts = contexts


class SimpleRAGPipeline:
    def __init__(self, retrieval, llm, top_k=3):  # Reduced default from 5 to 3
        self.retrieval = retrieval
        self.llm = llm
        self.top_k = top_k

    def run(self, query: str):
        print(f"[PIPELINE] Starting RAG run with query: {query}", flush=True)
        
        # Retrieve documents
        print(f"[PIPELINE] Retrieving {self.top_k} documents...", flush=True)
        docs = self.retrieval.retrieve(query, self.top_k)
        print(f"[PIPELINE] Retrieved {len(docs)} documents", flush=True)
        
        for i, doc in enumerate(docs):
            print(f"[PIPELINE] Context {i+1} length: {len(doc)}", flush=True)

        context = "\n---\n".join(docs)
        print(f"[PIPELINE] Total context length: {len(context)}", flush=True)

        # Optimized prompt for faster generation
        prompt = f"""Answer the question using ONLY the context below. Be concise (1-2 sentences).

Context:
{context}

Question: {query}
Answer:"""

        print(f"[PIPELINE] Prompt length: {len(prompt)}", flush=True)
        print("[PIPELINE] Calling LLM generate...", flush=True)
        
        answer = self.llm.generate(prompt)
        
        print(f"[PIPELINE] LLM returned answer length: {len(answer)}", flush=True)
        print(f"[PIPELINE] Answer preview: {answer[:100]}", flush=True)
        
        if not answer or not answer.strip():
            print("[PIPELINE] WARNING: Empty answer returned by LLM!", flush=True)

        return Answer(answer, docs)