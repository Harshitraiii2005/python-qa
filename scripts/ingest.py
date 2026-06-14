import logging
from pathlib import Path
import os

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def download_dataset():
    load_dotenv()

    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if not username or not key:
        raise ValueError(
            "KAGGLE_USERNAME and KAGGLE_KEY must be set in .env"
        )

    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    try:
        import kaggle
    except ImportError:
        raise ImportError(
            "Install Kaggle package:\n"
            "pip install kaggle"
        )

    logger.info("Authenticating with Kaggle...")
    kaggle.api.authenticate()

    logger.info("Downloading dataset into %s", data_dir)

    kaggle.api.dataset_download_files(
        "stackoverflow/pythonquestions",
        path=str(data_dir),
        unzip=True,
        quiet=False,
    )

    logger.info("Download completed.")

    logger.info("Files in data directory:")
    for file in sorted(data_dir.iterdir()):
        logger.info(" - %s", file.name)


if __name__ == "__main__":
    download_dataset()