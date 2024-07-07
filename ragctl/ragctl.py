"""This module provides the RAGCTL model-controller"""
# ragctl/ragctl.py

from pathlib import Path
from typing import Any, Dict, NamedTuple, List
from ragctl import DB_READ_ERROR, DOC_PATH_ERROR
from ragctl.model import DatabaseHandler
import os
import random

class CurrentDoc(NamedTuple):
    rag: Dict[str, Any]
    error: int

class RagDocer:
    def __init__(self, db_path: Path) -> None:
        self._db_handler = DatabaseHandler(db_path)
    
    def upload_doc(self, doc_paths: List[str]) -> CurrentDoc:
        doc_info_list = []
        """Add a new uploaded doc to the database"""
        # Check if file path in the list exists
        for doc_path in doc_paths:
            doc_info_dict = {}
            if not os.path.exists(doc_path):
                # Return an error if the file path does not exist.
                return CurrentDoc({}, DOC_PATH_ERROR)
            # Generate a random 4 digit number for the document
            doc_id = self._generate_doc_id()
            # Document name
            doc_name = os.path.basename(doc_path)
            # Document size
            doc_size = self._get_documents_size(doc_path)
            doc_info_dict = {
                "id": doc_id,
                "name": doc_name,
                "size": doc_size,
                "embeded": False
            }
            read = self._db_handler.read_ragdocs()
            if read.error == DB_READ_ERROR:
                return CurrentDoc(doc_info_dict, read.error)
            read.ragdoc_list.append(doc_info_dict)
            write = self._db_handler.write_ragdocs(read.ragdoc_list)
            if write.error:
                return CurrentDoc(doc_info_dict, write.error)
        return CurrentDoc(doc_info_dict, write.error)  
    
    # Generate random 4 digit number for the document
    def _generate_doc_id(self) -> int:
        return random.randint(1000, 9999)
    
    # Get the document size
    def _get_documents_size(self, document: str) -> str:
        # Get the file size in bytes
        size_in_bytes = os.path.getsize(document)
        # Determine the appropriate unit (KB or MB) based on file size
        if size_in_bytes < 1024:
            file_size = f'{size_in_bytes} bytes'
        elif size_in_bytes < (1024 * 1024):
            file_size = f'{round(size_in_bytes / 1024, 2)} KB'
        else:
            file_size = f"{size_in_bytes / (1024 * 1024):.2f} MB"
        return file_size
    
    # Get the list of documents
    def get_documents_list(self) -> List[Dict[str, Any]]:
        """Return the list of uploaded documents"""
        read = self._db_handler.read_ragdocs()
        return read.ragdoc_list
    
    # Clear all the documents from the database
    def clear_all(self) -> None:
        """Clear all the documents from the database"""
        write = self._db_handler.write_ragdocs([])
        return CurrentDoc({}, write.error)