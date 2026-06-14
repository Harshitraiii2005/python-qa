from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Groq
    groq_api_key: str = ""
    llm_model: str = "llama-3.1-8b-instant"
    llm_max_tokens: int = 1024

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Vector store
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "python_stackoverflow"
    top_k: int = 5

    # Data
    data_path: str = "./data/python_qa_sample.csv"
    max_documents: int = 10000

    # Kaggle credentials (for dataset download)
    kaggle_username: str = ""
    kaggle_key: str = ""

    # Google Drive (for chroma_db persistence)
    # Set GDRIVE_FOLDER_ID to your Google Drive folder ID
    # Set GDRIVE_CREDENTIALS_JSON to the path of your service account JSON
    gdrive_folder_id: str = ""
    gdrive_credentials_json: str = "./gdrive_credentials.json"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()