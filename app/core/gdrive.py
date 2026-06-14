from __future__ import annotations

import base64
import io
import json
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CHROMA_ZIP_NAME = "chroma_db.zip"


def _get_drive_service():
    """Build a Google Drive API service object."""
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError:
        raise ImportError(
            "google-api-python-client and google-auth are required.\n"
            "Run: pip install google-api-python-client google-auth"
        )

    from app.core.config import settings

    scopes = ["https://www.googleapis.com/auth/drive"]

    # Support base64-encoded credentials for Render (no file system secret)
    creds_b64 = os.getenv("GDRIVE_CREDENTIALS_B64", "")
    if creds_b64:
        creds_info = json.loads(base64.b64decode(creds_b64).decode())
        creds = service_account.Credentials.from_service_account_info(
            creds_info, scopes=scopes
        )
    else:
        creds_path = settings.gdrive_credentials_json
        if not os.path.exists(creds_path):
            raise FileNotFoundError(
                f"Google Drive credentials not found at {creds_path}.\n"
                "Set GDRIVE_CREDENTIALS_JSON or GDRIVE_CREDENTIALS_B64."
            )
        creds = service_account.Credentials.from_service_account_file(
            creds_path, scopes=scopes
        )

    return build("drive", "v3", credentials=creds)


def _find_file_id(service, folder_id: str, filename: str) -> Optional[str]:
    """Return the Drive file ID of filename inside folder_id, or None."""
    query = (
        f"name='{filename}' and "
        f"'{folder_id}' in parents and "
        "trashed=false"
    )
    result = service.files().list(q=query, fields="files(id,name)").execute()
    files = result.get("files", [])
    return files[0]["id"] if files else None


def download_chroma_from_drive(chroma_dir: str) -> bool:
    """
    Download chroma_db.zip from Drive and unzip to chroma_dir.
    Returns True if successful, False if not found or Drive not configured.
    """
    from app.core.config import settings

    folder_id = settings.gdrive_folder_id
    if not folder_id:
        logger.info("GDRIVE_FOLDER_ID not set — skipping Drive download.")
        return False

    try:
        from googleapiclient.http import MediaIoBaseDownload

        service = _get_drive_service()
        file_id = _find_file_id(service, folder_id, CHROMA_ZIP_NAME)

        if not file_id:
            logger.info("chroma_db.zip not found on Drive — will build from scratch.")
            return False

        logger.info("Found chroma_db.zip on Drive (id=%s) — downloading…", file_id)
        request = service.files().get_media(fileId=file_id)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        # Remove old chroma_dir if exists
        if os.path.exists(chroma_dir):
            shutil.rmtree(chroma_dir)

        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zf:
            # Extract into parent of chroma_dir so it becomes chroma_dir itself
            parent = str(Path(chroma_dir).parent)
            zf.extractall(parent)

        logger.info("chroma_db restored from Drive to %s", chroma_dir)
        return True

    except Exception as e:
        logger.warning("Drive download failed (%s) — will rebuild.", e)
        return False


def upload_chroma_to_drive(chroma_dir: str) -> bool:
    """
    Zip chroma_dir and upload to Drive, replacing any existing file.
    Returns True if successful.
    """
    from app.core.config import settings

    folder_id = settings.gdrive_folder_id
    if not folder_id:
        logger.info("GDRIVE_FOLDER_ID not set — skipping Drive upload.")
        return False

    if not os.path.exists(chroma_dir):
        logger.warning("chroma_dir %s does not exist — skipping upload.", chroma_dir)
        return False

    try:
        from googleapiclient.http import MediaIoBaseUpload

        logger.info("Zipping %s for Drive upload…", chroma_dir)
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(chroma_dir):
                for file in files:
                    abs_path = os.path.join(root, file)
                    arcname = os.path.relpath(abs_path, start=str(Path(chroma_dir).parent))
                    zf.write(abs_path, arcname)
        zip_buf.seek(0)
        size_mb = zip_buf.getbuffer().nbytes / 1024 / 1024
        logger.info("Zip size: %.1f MB", size_mb)

        service = _get_drive_service()

        # Delete existing file if present
        existing_id = _find_file_id(service, folder_id, CHROMA_ZIP_NAME)
        if existing_id:
            logger.info("Deleting old chroma_db.zip (id=%s) from Drive…", existing_id)
            service.files().delete(fileId=existing_id).execute()

        # Upload new zip
        file_metadata = {"name": CHROMA_ZIP_NAME, "parents": [folder_id]}
        media = MediaIoBaseUpload(zip_buf, mimetype="application/zip", resumable=True)
        uploaded = (
            service.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )
        logger.info(
            "chroma_db.zip uploaded to Drive successfully (id=%s)", uploaded.get("id")
        )
        return True

    except Exception as e:
        logger.error("Drive upload failed: %s", e)
        return False