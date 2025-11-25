import asyncio
from ragas.llms import BaseRagasLLM
from ragas.embeddings import BaseRagasEmbeddings
from app.models.language_model import query_google_llm

from langchain_core.outputs.llm_result import LLMResult  
from langchain_core.outputs.generation import Generation  



class GeminiRagasLLM(BaseRagasLLM):
    """
    RAGAS-compatible Gemini LLM wrapper using query_google_llm.
    Accepts arbitrary kwargs to be compatible with RAGAS calling conventions.
    Adds a 7-second delay to avoid hitting Gemini API quota limits.
    """

    def generate_text(self, prompt: str, *args, **kwargs) -> LLMResult:
        # synchronous throttling
        import time
        time.sleep(7)
        
        text = query_google_llm(
            db_answer="",
            vector_context="",
            user_query=prompt
        )

        # wrap in LLMResult with Generation objects for RAGAS
        return LLMResult(generations=[[Generation(text=text)]])

    async def agenerate_text(self, prompt: str, *args, **kwargs) -> LLMResult:
        # async-friendly throttling
        await asyncio.sleep(7)
        # reuse synchronous generate_text
        return self.generate_text(prompt, *args, **kwargs)

    def is_finished(self, response, *args, **kwargs) -> bool:
        """
        RAGAS calls this method to check if generation is complete.
        Accepts extra args/kwargs for maximum compatibility.
        """
        return True


class MiniLMRagasEmbeddings(BaseRagasEmbeddings):
    """Wrapper to use your MiniLM embedding model inside RAGAS."""

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    # ----- required by RAGAS -----

    # Sync: embed a single query string
    def embed_query(self, text: str):
        return self.embedding_model.embed(text)

    # Sync: embed a list of documents
    def embed_documents(self, texts: list[str]):
        return [self.embedding_model.embed(t) for t in texts]

    # Async: embed a single query
    async def aembed_query(self, text: str):
        return self.embed_query(text)

    # Async: embed multiple documents
    async def aembed_documents(self, texts: list[str]):
        return self.embed_documents(texts)
