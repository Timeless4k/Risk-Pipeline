"""
File utility functions for RiskPipeline.
"""

import os
import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class FileUtils:
    """Utility class for file operations."""
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """
        Ensure a directory exists, create if it doesn't.
        
        Args:
            path: Directory path
            
        Returns:
            Path object
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> bool:
        """
        Save data to JSON file.
        
        Args:
            data: Data to save
            filepath: File path
            indent: JSON indentation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, default=str)
            
            logger.debug(f"Saved JSON to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving JSON to {filepath}: {str(e)}")
            return False
    
    @staticmethod
    def load_json(filepath: Union[str, Path]) -> Optional[Any]:
        """
        Load data from JSON file.
        
        Args:
            filepath: File path
            
        Returns:
            Loaded data or None if failed
        """
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                logger.warning(f"JSON file not found: {filepath}")
                return None
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.debug(f"Loaded JSON from {filepath}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading JSON from {filepath}: {str(e)}")
            return None
    
    @staticmethod
    def save_pickle(data: Any, filepath: Union[str, Path]) -> bool:
        """
        Save data to pickle file.
        
        Args:
            data: Data to save
            filepath: File path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            logger.debug(f"Saved pickle to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving pickle to {filepath}: {str(e)}")
            return False
    
    @staticmethod
    def load_pickle(filepath: Union[str, Path]) -> Optional[Any]:
        """
        Load data from pickle file.
        
        Args:
            filepath: File path
            
        Returns:
            Loaded data or None if failed
        """
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                logger.warning(f"Pickle file not found: {filepath}")
                return None
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            logger.debug(f"Loaded pickle from {filepath}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading pickle from {filepath}: {str(e)}")
            return None
    
    @staticmethod
    def list_files(directory: Union[str, Path], 
                   pattern: str = "*", 
                   recursive: bool = False) -> List[Path]:
        """
        List files in directory matching pattern.
        
        Args:
            directory: Directory path
            pattern: File pattern (e.g., "*.txt")
            recursive: Whether to search recursively
            
        Returns:
            List of file paths
        """
        try:
            directory = Path(directory)
            
            if not directory.exists():
                logger.warning(f"Directory not found: {directory}")
                return []
            
            if recursive:
                files = list(directory.rglob(pattern))
            else:
                files = list(directory.glob(pattern))
            
            # Filter out directories
            files = [f for f in files if f.is_file()]
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing files in {directory}: {str(e)}")
            return []
    
    @staticmethod
    def get_file_size(filepath: Union[str, Path]) -> Optional[int]:
        """
        Get file size in bytes.
        
        Args:
            filepath: File path
            
        Returns:
            File size in bytes or None if failed
        """
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                logger.warning(f"File not found: {filepath}")
                return None
            
            return filepath.stat().st_size
            
        except Exception as e:
            logger.error(f"Error getting file size for {filepath}: {str(e)}")
            return None
    
    @staticmethod
    def copy_file(source: Union[str, Path], 
                  destination: Union[str, Path], 
                  overwrite: bool = False) -> bool:
        """
        Copy a file.
        
        Args:
            source: Source file path
            destination: Destination file path
            overwrite: Whether to overwrite existing file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            source = Path(source)
            destination = Path(destination)
            
            if not source.exists():
                logger.error(f"Source file not found: {source}")
                return False
            
            if destination.exists() and not overwrite:
                logger.warning(f"Destination file exists and overwrite=False: {destination}")
                return False
            
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            
            logger.debug(f"Copied {source} to {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Error copying {source} to {destination}: {str(e)}")
            return False
    
    @staticmethod
    def delete_file(filepath: Union[str, Path]) -> bool:
        """
        Delete a file.
        
        Args:
            filepath: File path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                logger.warning(f"File not found: {filepath}")
                return False
            
            filepath.unlink()
            logger.debug(f"Deleted file: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file {filepath}: {str(e)}")
            return False
    
    @staticmethod
    def delete_directory(directory: Union[str, Path], 
                        recursive: bool = False) -> bool:
        """
        Delete a directory.
        
        Args:
            directory: Directory path
            recursive: Whether to delete recursively
            
        Returns:
            True if successful, False otherwise
        """
        try:
            directory = Path(directory)
            
            if not directory.exists():
                logger.warning(f"Directory not found: {directory}")
                return False
            
            if recursive:
                shutil.rmtree(directory)
            else:
                directory.rmdir()
            
            logger.debug(f"Deleted directory: {directory}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting directory {directory}: {str(e)}")
            return False
    
    @staticmethod
    def get_file_extension(filepath: Union[str, Path]) -> str:
        """
        Get file extension.
        
        Args:
            filepath: File path
            
        Returns:
            File extension (without dot)
        """
        return Path(filepath).suffix.lstrip('.')
    
    @staticmethod
    def get_filename_without_extension(filepath: Union[str, Path]) -> str:
        """
        Get filename without extension.
        
        Args:
            filepath: File path
            
        Returns:
            Filename without extension
        """
        return Path(filepath).stem
    
    @staticmethod
    def create_backup(filepath: Union[str, Path], 
                     backup_suffix: str = ".backup") -> Optional[Path]:
        """
        Create a backup of a file.
        
        Args:
            filepath: File path
            backup_suffix: Suffix for backup file
            
        Returns:
            Backup file path or None if failed
        """
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                logger.warning(f"File not found: {filepath}")
                return None
            
            backup_path = filepath.with_suffix(filepath.suffix + backup_suffix)
            FileUtils.copy_file(filepath, backup_path, overwrite=True)
            
            logger.debug(f"Created backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Error creating backup for {filepath}: {str(e)}")
            return None 