"""This module provides the RAGCTL model-controller"""
# ragctl/ragctl.py

from pathlib import Path
from typing import Any, Dict, NamedTuple, List
from ragctl import DB_READ_ERROR, DOC_PATH_ERROR, DOC_DUPLICATE_ERROR, ID_ERROR, SUCCESS
from ragctl.model import DatabaseHandler
import os
import random
import shutil
import hashlib

class CurrentDoc(NamedTuple):
    rag: Dict[str, Any]
    error: int

class RagDocer:
    def __init__(self, db_path: Path) -> None:
        self._db_handler = DatabaseHandler(db_path)
        # Set the data folder path
        self.data_folder = Path(__file__).parent / "data"
    
    def upload_doc(self, doc_paths: List[str]) -> CurrentDoc:
        try:
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
                # Document MD5SUM
                md5sum = self._calculate_md5sum(doc_path)
                # Check if md5sum of the document already exists
                if md5sum in [doc["md5sum"] for doc in self.get_documents_list()]:
                    return CurrentDoc({}, DOC_DUPLICATE_ERROR)
                doc_info_dict = {
                    "id": doc_id,
                    "name": doc_name,
                    "size": doc_size,
                    "embedded": "False",
                    "md5sum": md5sum
                }
                read = self._db_handler.read_ragdocs()
                if read.error == DB_READ_ERROR:
                    return CurrentDoc(doc_info_dict, read.error)
                read.ragdoc_list.append(doc_info_dict)
                write = self._db_handler.write_ragdocs(read.ragdoc_list)
                if write.error:
                    return CurrentDoc(doc_info_dict, write.error)
                # Copy the document to the data folder
                shutil.copy(doc_path, self.data_folder / doc_name)
            return CurrentDoc(doc_info_dict, write.error)
        except Exception as error:
            return CurrentDoc({}, error) 
    
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
    
    # Calculate the MD5SUM of the file
    def _calculate_md5sum(self, file_path: str) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    # Get the list of documents
    def get_documents_list(self) -> List[Dict[str, Any]]:
        """Return the list of uploaded documents"""
        read = self._db_handler.read_ragdocs()
        return read.ragdoc_list
    
    # Clear all the documents from the database
    def clear_all(self) -> None:
        """Clear all the documents from the database"""
        write = self._db_handler.write_ragdocs([])
        # Delete all the documents from data folder path except README.md
        for file in os.listdir(self.data_folder):
            if file != "README.md":
                os.remove(self.data_folder / file)
        return CurrentDoc({}, write.error)
    
    # Get the list of documents which are not embedded
    def get_non_embedded_documents(self) -> List[Dict[str, Any]]:
        """Return the list of non-embedded documents"""
        read = self._db_handler.read_ragdocs()
        if read.error == DB_READ_ERROR:
            return []
        return [doc for doc in read.ragdoc_list if doc["embedded"] == "False"]
    
    # Get the list of documents which are embedded
    def get_embedded_documents(self) -> List[Dict[str, Any]]:
        """Return the list of embedded documents"""
        read = self._db_handler.read_ragdocs()
        if read.error == DB_READ_ERROR:
            return []
        return [doc for doc in read.ragdoc_list if doc["embedded"] == "True"]
    
    # Perform embedding on a document
    def embed_document(self, doc_id: str):
        """Embed a document"""
        read = self._db_handler.read_ragdocs()
        if read.error == DB_READ_ERROR:
            return []
        
        # Check if the document id exists
        if not any(doc["id"] == doc_id for doc in read.ragdoc_list):
            return CurrentDoc({}, ID_ERROR)
        else:
            return CurrentDoc({}, SUCCESS)