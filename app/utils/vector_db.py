# vector_db.py

import os
from langchain_core.documents import Document
from app.utils.custom_logging import logger
from app.utils.string_to_dict_parser import parse_row_to_metadata
from pinecone import Pinecone, ServerlessSpec


class PineconeVectorDB:
    """
    Dense-only Pinecone vector DB client.
    Supports:
    - Multiple indexes
    - Different embedding models
    - Upsert + Query
    """

    def __init__(
        self,
        api_key: str = None,
        index_name: str = "default-index",
        dimension: int = 1536,
        region: str = "us-east-1",
        cloud: str = "aws",
        embedding_model=None
    ):
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("Missing Pinecone API Key.")

        self.pc = Pinecone(api_key=self.api_key)

        self.index_name = index_name
        self.dimension = dimension
        self.region = region
        self.cloud = cloud
        self.embeddings = embedding_model

        # Create if missing
        self._create_index_if_not_exists()

        # Load index
        self.index = self.pc.Index(self.index_name)

    def _create_index_if_not_exists(self):
        existing = [idx["name"] for idx in self.pc.list_indexes()]
        if self.index_name not in existing:
            logger.info(f"[Pinecone] Creating index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region)
            )
        else:
            logger.info(f"[Pinecone] Index '{self.index_name}' already exists.")

    # -------------------------------------------------------------
    # UPSERT PIPELINE
    # -------------------------------------------------------------
    def upsert_documents(self, series):
        logger.info(f"[Pinecone] Preparing {len(series)} documents...")

        documents = []
        ids = []
        texts_for_embedding = []

        # 1. Parse + build Documents
        for text in series:
            metadata = parse_row_to_metadata(text)
            doc_id = metadata.get("setId")
            if not doc_id:
                raise ValueError(f"[Error] Missing setId in metadata: {metadata}")

            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
            ids.append(str(doc_id))
            texts_for_embedding.append(text)

        # 2. Embeddings
        logger.info("[Pinecone] Generating embeddings...")
        embeddings = self.embeddings.embed_documents(texts_for_embedding)

        if len(embeddings) != len(documents):
            raise RuntimeError("Embedding count mismatch!")

        # 3. Pinecone format
        vectors_payload = []
        for doc, emb, doc_id in zip(documents, embeddings, ids):
            vectors_payload.append({
                "id": doc_id,
                "values": emb,
                "metadata": {
                    **doc.metadata,
                    "text": doc.page_content   # <-- store full chunk text
                }
            })

        # 4. Upsert
        logger.info(f"[Pinecone] Upserting {len(vectors_payload)} vectors...")
        return self.index.upsert(vectors=vectors_payload)

    # -------------------------------------------------------------
    # QUERY PIPELINE
    # -------------------------------------------------------------
    def query(self, query_text: str, k: int = 5):
        print("[Pinecone] Embedding query...")
        query_vector = self.embeddings.embed_query(query_text)

        # Convert ndarray to list for Pinecone
        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()

        result = self.index.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True
        )

        if not result.matches:
            return "No vector matches found."

        blocks = []
        for i, match in enumerate(result.matches, start=1):
            meta = match.metadata  # this is already a dict
            chunk = meta.get("text", "")
            score = match.score

            meta_lines = "\n".join([f"   - {key}: {value}" for key, value in meta.items() if key != "text"])

            block = f"""
        {i}) Chunk:
        {chunk}

        Metadata:
        {meta_lines}

        Score: {score:.3f}
        """
        blocks.append(block.strip())

        return "\n\n".join(blocks)
