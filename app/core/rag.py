from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from pathlib import Path
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
        """Full startup: restore or build index, then mark ready."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_init)

    def _sync_init(self):
        logger.info("=" * 60)
        logger.info("RAG Pipeline startup")
        logger.info("=" * 60)
        t0 = time.time()

        # Step 1 — embedding function (local, no API needed)
        logger.info("[1/5] Loading embedding model: %s", settings.embedding_model)
        self._embed_fn = SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )

        # Step 2 — try to restore chroma_db from Google Drive
        logger.info("[2/5] Checking Google Drive for existing index…")
        restored = self._try_restore_from_drive()

        # Step 3 — open / create the ChromaDB collection
        logger.info("[3/5] Opening ChromaDB at %s", settings.chroma_persist_dir)
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # Step 4 — ingest if collection is empty (first run or Drive unavailable)
        if self._collection.count() == 0:
            logger.info("[4/5] Collection empty — building index from scratch…")
            self._ensure_data()
            self._ingest()
            # Upload fresh index to Drive for next time
            self._upload_to_drive()
        else:
            logger.info(
                "[4/5] Collection has %d docs — skipping ingest.",
                self._collection.count(),
            )

        # Step 5 — Groq LLM client
        logger.info("[5/5] Initialising Groq LLM client…")
        api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY", "")
        if not api_key:
            logger.warning("GROQ_API_KEY not set — /ask will fail at generation step.")
        self._llm = Groq(api_key=api_key)

        self._ready = True
        logger.info("RAG pipeline ready in %.1fs  (%d docs)", time.time() - t0, self._collection.count())
        logger.info("=" * 60)

    # ------------------------------------------------------------------ #
    # Drive helpers                                                         #
    # ------------------------------------------------------------------ #

    def _try_restore_from_drive(self) -> bool:
        try:
            from app.core.gdrive import download_chroma_from_drive
            return download_chroma_from_drive(settings.chroma_persist_dir)
        except Exception as e:
            logger.warning("Drive restore skipped: %s", e)
            return False

    def _upload_to_drive(self):
        try:
            from app.core.gdrive import upload_chroma_to_drive
            upload_chroma_to_drive(settings.chroma_persist_dir)
        except Exception as e:
            logger.warning("Drive upload skipped: %s", e)

    # ------------------------------------------------------------------ #
    # Data acquisition                                                      #
    # ------------------------------------------------------------------ #

    def _ensure_data(self):
        """
        Make sure the processed CSV exists.
        If not, download from Kaggle and merge Questions + Answers.
        """
        csv_path = Path(settings.data_path)

        if csv_path.exists():
            logger.info("Data CSV already exists at %s — skipping download.", csv_path)
            return

        logger.info("Data CSV not found — downloading from Kaggle…")
        self._download_from_kaggle(csv_path)

    def _download_from_kaggle(self, out_csv: Path):
        """Download the Stack Overflow Python Questions dataset and merge CSVs."""
        # Set Kaggle credentials
        username = settings.kaggle_username or os.getenv("KAGGLE_USERNAME", "")
        key = settings.kaggle_key or os.getenv("KAGGLE_KEY", "")

        if not username or not key:
            raise RuntimeError(
                "KAGGLE_USERNAME and KAGGLE_KEY must be set to download the dataset."
            )

        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key

        try:
            import kaggle
        except ImportError:
            raise ImportError("Run: pip install kaggle")

        data_dir = out_csv.parent
        data_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Authenticating with Kaggle…")
        kaggle.api.authenticate()

        logger.info("Downloading stackoverflow/pythonquestions → %s", data_dir)
        kaggle.api.dataset_download_files(
            "stackoverflow/pythonquestions",
            path=str(data_dir),
            unzip=True,
            quiet=False,
        )
        logger.info("Download complete.")

        # Merge Questions + Answers
        self._merge_csvs(data_dir, out_csv)

    def _merge_csvs(self, data_dir: Path, out_csv: Path):
        """Merge Questions.csv + Answers.csv into a single enriched CSV."""
        q_path = data_dir / "Questions.csv"
        a_path = data_dir / "Answers.csv"

        logger.info("Loading Questions.csv…")
        questions = pd.read_csv(
            q_path, encoding="latin-1",
            usecols=["Id", "Title", "Body", "Score"]
        )
        questions = questions[questions["Score"] >= 1].dropna()

        if a_path.exists():
            logger.info("Loading Answers.csv and merging best answers…")
            answers = pd.read_csv(
                a_path, encoding="latin-1",
                usecols=["ParentId", "Body", "Score"]
            )
            best = (
                answers.sort_values("Score", ascending=False)
                .groupby("ParentId")
                .first()
                .reset_index()
                .rename(columns={"ParentId": "Id", "Body": "BestAnswer"})
            )
            df = questions.merge(best[["Id", "BestAnswer"]], on="Id", how="left")
            df["Body"] = df.apply(
                lambda r: r["Body"] + (
                    f"\n\nBest Answer:\n{r['BestAnswer']}"
                    if pd.notna(r.get("BestAnswer")) else ""
                ),
                axis=1,
            )
            df.drop(columns=["BestAnswer"], inplace=True, errors="ignore")
        else:
            df = questions

        df.to_csv(out_csv, index=False, encoding="utf-8")
        logger.info("Merged CSV saved → %s  (%d rows)", out_csv, len(df))

    # ------------------------------------------------------------------ #
    # Ingest                                                                #
    # ------------------------------------------------------------------ #

    def _ingest(self):
        """Read CSV, clean, and upsert into ChromaDB."""
        path = settings.data_path
        if not os.path.exists(path):
            logger.warning("Data file not found at %s — collection will be empty.", path)
            return

        logger.info("Loading %s (max %d rows)…", path, settings.max_documents)
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
        logger.info("Ingesting %d documents…", len(df))

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

        logger.info("Ingestion complete. Total docs: %d", self._collection.count())

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

        context = "\n\n---\n\n".join(
            f"[Source {i+1}] {doc}" for i, doc in enumerate(docs)
        )

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
            "question":   question,
            "answer":     answer,
            "sources":    sources,
            "latency_ms": round((time.time() - t0) * 1000),
            "model":      settings.llm_model,
        }

    # ------------------------------------------------------------------ #
    # Status                                                                #
    # ------------------------------------------------------------------ #

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
