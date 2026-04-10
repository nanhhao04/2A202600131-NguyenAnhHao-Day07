from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        # TODO: store references to store and llm_fn
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # TODO: retrieve chunks, build prompt, call llm_fn
        if not question:
            return ""
            
        chunks = self.store.search(question, top_k=top_k)
        
        context_parts = [c["content"] for c in chunks]
        context_text = "\n---\n".join(context_parts)
        
        prompt = (
            "You are a helpful assistant. Answer the question using ONLY the provided context. "
            "If the context doesn't contain the answer, say you don't know.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        
        return self.llm_fn(prompt)
