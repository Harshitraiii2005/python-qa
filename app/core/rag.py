"""
RAG Pipeline: ingest → embed → retrieve → generate
Uses ChromaDB (local) + sentence-transformers + Groq (free, fast)
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Optional

from groq import Groq
import chromadb
import pandas as pd
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from fastapi import HTTPException

from app.core.config import settings

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for Python Q&A."""

    def __init__(self):
        self._ready = False
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None
        self._llm: Optional[Groq] = None
        self._embed_fn = None

    # ------------------------------------------------------------------ #
    # Initialization                                                        #
    # ------------------------------------------------------------------ #

    async def initialize(self):
        """Load or build the vector store, then mark ready."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_init)

    def _sync_init(self):
        logger.info("Initialising RAG pipeline …")
        t0 = time.time()

        # Embedding function (runs locally, no API key needed)
        self._embed_fn = SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )

        # Vector store
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # Ingest data only when collection is empty
        if self._collection.count() == 0:
            logger.info("Collection empty — ingesting data …")
            self._ingest()
        else:
            logger.info("Collection has %d docs — skipping ingest.", self._collection.count())

        # Groq client
        api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY", "")
        if not api_key:
            logger.warning("GROQ_API_KEY not set — /ask will fail at generation step.")
        self._llm = Groq(api_key=api_key)

        self._ready = True
        logger.info("RAG pipeline ready in %.1fs", time.time() - t0)

    def _ingest(self):
        """Read CSV, clean, chunk, and upsert into ChromaDB."""
        path = settings.data_path
        if not os.path.exists(path):
            logger.warning("Data file not found at %s — collection will be empty.", path)
            return

        logger.info("Loading %s …", path)
        df = pd.read_csv(path, nrows=settings.max_documents)

        required = {"Title", "Body", "Score"}
        if not required.issubset(df.columns):
            if "Body" in df.columns:
                df["Title"] = df.get("Id", range(len(df))).astype(str)
                df["Score"] = df.get("Score", 0)
            else:
                logger.error("Unrecognised CSV schema: %s", df.columns.tolist())
                return

        df = df[df["Score"] >= 1].dropna(subset=["Body"]).reset_index(drop=True)
        df = df.head(settings.max_documents)

        logger.info("Ingesting %d documents …", len(df))

        documents, metadatas, ids = [], [], []
        for idx, row in df.iterrows():
            title = str(row.get("Title", "")).strip()
            body  = _clean_html(str(row.get("Body", "")))
            text  = f"Q: {title}\n\n{body}"[:2000]
            documents.append(text)
            metadatas.append({
                "title": title[:256],
                "score": int(row.get("Score", 0)),
                "id":    str(row.get("Id", idx)),
            })
            ids.append(str(row.get("Id", idx)))

        batch = 2000
        for start in range(0, len(documents), batch):
            end = start + batch
            self._collection.upsert(
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )
            logger.info("  upserted %d / %d", min(end, len(documents)), len(documents))

        logger.info("Ingestion complete. Total: %d", self._collection.count())

    # ------------------------------------------------------------------ #
    # Retrieval + Generation                                               #
    # ------------------------------------------------------------------ #

    async def ask(self, question: str) -> dict:
        if not self._ready:
            raise HTTPException(503, "RAG pipeline not ready yet — try again in a moment.")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_ask, question)

    def _sync_ask(self, question: str) -> dict:
        t0 = time.time()

        # --- Retrieve ---
        results = self._collection.query(
            query_texts=[question],
            n_results=settings.top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        sources = [
            {
                "title":     m.get("title", "")[:120],
                "score":     m.get("score", 0),
                "relevance": round(1 - d, 3),
                "so_id":     m.get("id", ""),
            }
            for m, d in zip(metas, distances)
        ]

        # --- Build context ---
        context = "\n\n---\n\n".join(
            f"[Source {i+1}] {doc}" for i, doc in enumerate(docs)
        )

        # --- Generate via Groq ---
        system = (
            "You are a Python programming tutor. "
            "Answer the user's question using ONLY the provided Stack Overflow context. "
            "Be precise, include code examples where relevant, and cite which source(s) you used "
            "by referencing [Source N]. "
            "If the context doesn't contain enough information, say so honestly."
        )
        prompt = f"""Context from Stack Overflow:
{context}

Question: {question}

Answer (cite sources):"""

        response = self._llm.chat.completions.create(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
        )
        answer = response.choices[0].message.content

        return {
            "question": question,
            "answer":   answer,
            "sources":  sources,
            "latency_ms": round((time.time() - t0) * 1000),
            "model":    settings.llm_model,
        }

    @property
    def ready(self) -> bool:
        return self._ready

    def stats(self) -> dict:
        if not self._ready:
            return {"status": "initialising"}
        return {
            "status":    "ready",
            "documents": self._collection.count() if self._collection else 0,
            "model":     settings.llm_model,
            "embedding": settings.embedding_model,
            "top_k":     settings.top_k,
        }


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Singleton
rag_pipeline = RAGPipeline()
