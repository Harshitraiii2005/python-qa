import argparse
import logging
import os
import sys
from pathlib import Path

# Make sure app package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build ChromaDB index for Python Q&A")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip Kaggle download (use existing CSV)")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Skip Google Drive upload after building index")
    parser.add_argument("--data-dir", default="data", help="Directory for CSVs")
    parser.add_argument("--out", default="data/python_qa_sample.csv",
                        help="Output merged CSV path")
    args = parser.parse_args()

    from app.core.config import settings

    out_csv = Path(args.out)

    # ── Step 1: Download dataset ──────────────────────────────────────────
    if not args.skip_download:
        if out_csv.exists():
            logger.info("CSV already exists at %s — skipping download.", out_csv)
        else:
            logger.info("=== Step 1: Downloading from Kaggle ===")
            username = settings.kaggle_username or os.getenv("KAGGLE_USERNAME", "")
            key = settings.kaggle_key or os.getenv("KAGGLE_KEY", "")

            if not username or not key:
                logger.error("Set KAGGLE_USERNAME and KAGGLE_KEY in .env")
                sys.exit(1)

            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"] = key

            import kaggle
            data_dir = Path(args.data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)

            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                "stackoverflow/pythonquestions",
                path=str(data_dir),
                unzip=True,
                quiet=False,
            )
            logger.info("Download complete.")

            # Merge
            import pandas as pd
            q_path = data_dir / "Questions.csv"
            a_path = data_dir / "Answers.csv"

            logger.info("Loading Questions.csv…")
            questions = pd.read_csv(q_path, encoding="latin-1",
                                    usecols=["Id", "Title", "Body", "Score"])
            questions = questions[questions["Score"] >= 1].dropna()

            if a_path.exists():
                logger.info("Merging Answers.csv…")
                answers = pd.read_csv(a_path, encoding="latin-1",
                                      usecols=["ParentId", "Body", "Score"])
                best = (
                    answers.sort_values("Score", ascending=False)
                    .groupby("ParentId").first().reset_index()
                    .rename(columns={"ParentId": "Id", "Body": "BestAnswer"})
                )
                df = questions.merge(best[["Id", "BestAnswer"]], on="Id", how="left")
                df["Body"] = df.apply(
                    lambda r: r["Body"] + (
                        f"\n\nBest Answer:\n{r['BestAnswer']}"
                        if pd.notna(r.get("BestAnswer")) else ""
                    ), axis=1,
                )
                df.drop(columns=["BestAnswer"], inplace=True, errors="ignore")
            else:
                df = questions

            df.to_csv(out_csv, index=False, encoding="utf-8")
            logger.info("Saved %d rows → %s", len(df), out_csv)
    else:
        logger.info("Skipping download (--skip-download set)")

    # ── Step 2: Build ChromaDB ────────────────────────────────────────────
    logger.info("=== Step 2: Building ChromaDB index ===")
    import shutil
    import chromadb
    import pandas as pd
    import re
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    chroma_dir = settings.chroma_persist_dir
    if os.path.exists(chroma_dir):
        logger.info("Removing existing chroma_db at %s", chroma_dir)
        shutil.rmtree(chroma_dir)

    embed_fn = SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(
        name=settings.collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    df = pd.read_csv(str(out_csv), nrows=settings.max_documents)
    df = df[df["Score"] >= 1].dropna(subset=["Body"]).reset_index(drop=True)
    df = df.head(settings.max_documents)
    logger.info("Ingesting %d documents…", len(df))

    def clean_html(text):
        text = re.sub(r"<[^>]+>", " ", str(text))
        text = re.sub(r"&\w+;", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    documents, metadatas, ids = [], [], []
    for idx, row in df.iterrows():
        title = str(row.get("Title", "")).strip()
        body  = clean_html(row.get("Body", ""))
        text  = f"Q: {title}\n\n{body}"[:2000]
        documents.append(text)
        metadatas.append({"title": title[:256], "score": int(row.get("Score", 0)),
                          "id": str(row.get("Id", idx))})
        ids.append(str(row.get("Id", idx)))

    batch = 2000
    for start in range(0, len(documents), batch):
        end = start + batch
        collection.upsert(documents=documents[start:end],
                          metadatas=metadatas[start:end], ids=ids[start:end])
        logger.info("  upserted %d / %d", min(end, len(documents)), len(documents))

    logger.info("Index built. Total docs: %d", collection.count())

    # ── Step 3: Upload to Google Drive ────────────────────────────────────
    if not args.skip_upload:
        logger.info("=== Step 3: Uploading chroma_db to Google Drive ===")
        try:
            from app.core.gdrive import upload_chroma_to_drive
            ok = upload_chroma_to_drive(chroma_dir)
            if ok:
                logger.info("Upload successful — future deploys will restore from Drive.")
            else:
                logger.warning("Upload skipped (GDRIVE_FOLDER_ID not set?).")
        except Exception as e:
            logger.warning("Upload failed: %s", e)
    else:
        logger.info("Skipping Drive upload (--skip-upload set)")

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()