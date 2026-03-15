"""
Data Loader - Handle file uploads and load data into pandas DataFrames

Supports multiple file formats:
- CSV (.csv)
- Excel (.xlsx, .xls)
"""

# Add project root to path
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Tuple
import logging
from Config.config import Config

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads data from various file formats into pandas DataFrame.

    Handles:
    - File type detection
    - Encoding detection
    - Error handling
    - Memory-efficient loading
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.allowed_extensions = self.config.ALLOWED_EXTENSIONS

    def load(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from file into DataFrame.

        Returns:
            pandas DataFrame with loaded data
        """

        # Convert to Path object
        file_path = Path(file_path)

        # Check file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file extension
        extension = file_path.suffix.lower()

        # Check if supported
        if extension not in self.allowed_extensions:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {self.allowed_extensions}"
            )

        logger.info(f"Loading file: {file_path.name} (type: {extension})")

        # Load based on file type
        try:
            if extension == '.csv':
                df = self._load_csv(file_path)
            elif extension in ['.xlsx', '.xls']:
                df = self._load_excel(file_path)
            else:
                raise ValueError(f"Handler not implemented for {extension}")

            logger.info(f"✅ Loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df

        except Exception as e:
            logger.error(f"❌ Failed to load file: {e}")
            raise

    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Load CSV file with automatic encoding detection.

        Tries multiple encoding strategies:
        1. UTF-8 (most common)
        2. Latin-1 (Western European)
        3. Auto-detect with chardet
        """
        # Try UTF-8 first (most common)
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            logger.debug("Loaded with UTF-8 encoding")
            return df
        except UnicodeDecodeError:
            logger.debug("UTF-8 failed, trying latin-1...")

        # Try latin-1 (handles most Western European files)
        try:
            df = pd.read_csv(file_path, encoding='latin-1')
            logger.debug("Loaded with latin-1 encoding")
            return df
        except UnicodeDecodeError:
            logger.debug("latin-1 failed, trying ISO-8859-1...")

        # Try ISO-8859-1 as fallback
        try:
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
            logger.debug("Loaded with ISO-8859-1 encoding")
            return df
        except Exception as e:
            raise ValueError(f"Could not determine file encoding. Error: {e}")

    def _load_excel(self, file_path: Path) -> pd.DataFrame:
        try:
            # Load first sheet by default
            df = pd.read_excel(file_path, sheet_name=0, engine='openpyxl' if file_path.suffix == '.xlsx' else None)
            logger.debug(f"Loaded first sheet from Excel file")
            return df
        except Exception as e:
            raise ValueError(f"Could not load Excel file: {e}")

    def load_from_streamlit_upload(self, uploaded_file) -> pd.DataFrame:
        """
        Load data from Streamlit file uploader object.
        Returns:
            pandas DataFrame
        """
        if uploaded_file is None:
            raise ValueError("No file uploaded")

        # Get file extension from name
        file_name = uploaded_file.name
        extension = Path(file_name).suffix.lower()

        # Check if supported
        if extension not in self.allowed_extensions:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {self.allowed_extensions}"
            )

        logger.info(f"Loading uploaded file: {file_name}")

        try:
            if extension == '.csv':
                # Try different encodings
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)  # Reset file pointer
                    df = pd.read_csv(uploaded_file, encoding='latin-1')

            elif extension in ['.xlsx', '.xls']:
                df = pd.read_excel(uploaded_file, sheet_name=0)
            else:
                raise ValueError(f"Handler not implemented for {extension}")

            logger.info(f"✅ Loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df

        except Exception as e:
            logger.error(f"❌ Failed to load uploaded file: {e}")
            raise

    def save_uploaded_file(self, uploaded_file, save_dir: Path = None) -> Path:
        """
        Save Streamlit uploaded file to disk.

        Returns:
            Path to saved file
        """
        if save_dir is None:
            save_dir = self.config.UPLOADS_DIR

        # Create directory if doesn't exist
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create unique filename (add timestamp to avoid overwrites)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = Path(uploaded_file.name)
        unique_name = f"{original_name.stem}_{timestamp}{original_name.suffix}"
        save_path = save_dir / unique_name

        # Save file
        with open(save_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        logger.info(f"💾 Saved uploaded file to: {save_path}")
        return save_path

    def get_file_info(self, file_path: Union[str, Path]) -> dict:
        """
        Get metadata about a file.

        Returns:
            Dictionary with file information:
            - name: filename
            - size_mb: file size in MB
            - extension: file type
            - exists: whether file exists
        """
        file_path = Path(file_path)

        info = {
            'name': file_path.name,
            'extension': file_path.suffix.lower(),
            'exists': file_path.exists()
        }

        if file_path.exists():
            size_bytes = file_path.stat().st_size
            info['size_mb'] = round(size_bytes / (1024 * 1024), 2)
        else:
            info['size_mb'] = 0

        return info

    def preview_file(self, file_path: Union[str, Path], n_rows: int = 5) -> pd.DataFrame:
        """
        Load just first few rows for preview (memory efficient).
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        if extension == '.csv':
            return pd.read_csv(file_path, nrows=n_rows)
        elif extension in ['.xlsx', '.xls']:
            return pd.read_excel(file_path, nrows=n_rows)
        else:
            raise ValueError(f"Preview not supported for {extension}")


# Convenience function
def load_data(file_path: Union[str, Path], config: Config = None) -> pd.DataFrame:
    """
    Quick function to load data without creating loader instance.

    Returns:
        pandas DataFrame
    """
    loader = DataLoader(config)
    return loader.load(file_path)