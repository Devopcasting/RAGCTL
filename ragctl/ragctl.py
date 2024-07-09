"""This module provides the RAGCTL model-controller"""
# ragctl/ragctl.py

from pathlib import Path
from typing import Any, Dict, NamedTuple, List
from ragctl import DB_READ_ERROR, DOC_PATH_ERROR, DOC_DUPLICATE_ERROR, ID_ERROR, SUCCESS, DOC_ALREADY_EMBEDDED, INVALID_PDF_FILE, EMBEDDING_ERROR
from ragctl.model import DatabaseHandler
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain.schema.document import Document
from langchain.prompts import ChatPromptTemplate
import os
import random
import shutil
import hashlib
import PyPDF2

class CurrentDoc(NamedTuple):
    rag: Dict[str, Any]
    error: int

class RagDocer:
    def __init__(self, db_path: Path) -> None:
        self._db_handler = DatabaseHandler(db_path)
        # Set the data folder path
        self.data_folder = Path(__file__).parent / "data"
        # Set the vectordb folder path
        self.vectordb_folder = Path(__file__).parent / "vectordb"
    
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
                # Check if document is a valid pdf file
                if not self._is_valid_pdf(doc_path):
                    return CurrentDoc({}, INVALID_PDF_FILE)
                
                doc_info_dict = {
                    "id": doc_id,
                    "name": doc_name,
                    "size": doc_size,
                    "embedded": "False",
                    "md5sum": md5sum,
                    "path": f"{self.data_folder}/{doc_id}/{doc_name}"
                }
                read = self._db_handler.read_ragdocs()
                if read.error == DB_READ_ERROR:
                    return CurrentDoc(doc_info_dict, read.error)
                read.ragdoc_list.append(doc_info_dict)
                write = self._db_handler.write_ragdocs(read.ragdoc_list)
                if write.error:
                    return CurrentDoc(doc_info_dict, write.error)
                
                # Create the id folder inside data
                if not os.path.exists(self.data_folder / str(doc_id)):
                    os.makedirs(self.data_folder / str(doc_id))

                # Copy the document to the id folder inside data
                shutil.copy(doc_path, self.data_folder / str(doc_id))
            #return CurrentDoc(doc_info_dict, write.error)
            return CurrentDoc(doc_info_dict, SUCCESS)
        except Exception as error:
            return CurrentDoc({}, error) 
    
    # Check if the document is valid PDF
    def _is_valid_pdf(self, doc_path: str) -> bool:
        """Check if the document is a valid PDF"""
        try:
            with open(doc_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                if len(pdf_reader.pages) == 0:
                    return False
                return True
        except PyPDF2.errors.PdfReadError:
            return False
    
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
        # Delete all the id directories inside data folder path except README.md
        for file in os.listdir(self.data_folder):
            if file != "README.md":
                shutil.rmtree(self.data_folder / file)
        
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
    def embed_document(self, doc_id: int) -> CurrentDoc:
        try:
            """Embed a document"""
            read = self._db_handler.read_ragdocs()

            if read.error == DB_READ_ERROR:
                return []
        
            # Check if the document id already exists
            doc_id_found = False
            for doc in read.ragdoc_list:
                if doc["id"] == doc_id:
                    doc_id_found = True
                    break
            if not doc_id_found:
                return CurrentDoc({}, ID_ERROR)
        
            # Check if the document is already embedded
            if doc["embedded"] == "True":
                return CurrentDoc(doc, DOC_ALREADY_EMBEDDED)
        
            # Load the PDF document
            doc_path = f"{self.data_folder}/{str(doc_id)}/{doc['name']}"
            pages = self._load_pdf_document(doc_path)

            # Split the PDF document into chunks
            chunks = self._split_documents(pages)

            # Add the PDF data to Chroma DB
            self._add_pdf_data_to_chroma(chunks, f"{self.vectordb_folder}")

            # Change the embedded status to True
            doc["embedded"] = "True"
            write = self._db_handler.write_ragdocs(read.ragdoc_list)
            if write.error:
                return CurrentDoc(doc, write.error)
            return CurrentDoc(doc, SUCCESS)
        except Exception as error:
            return CurrentDoc({}, EMBEDDING_ERROR)
    
    # Load PDF document
    def _load_pdf_document(self, doc_path: str):
        """Load a PDF document"""
        loader = PyPDFLoader(doc_path)
        pages = loader.load()
        return pages
    
    # Split PDF documents
    def _split_documents(self, pages: list[Document], chunk_size=800, chunk_overlap=80):
        """Split a PDF document into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        return text_splitter.split_documents(pages)
    
    # Add PDF data to Chroma DB
    def _add_pdf_data_to_chroma(self, chunks: list[Document], vectordb_path: str):
        """Add PDF data to Chroma DB"""
        db = Chroma(
            embedding_function=self._aws_bedrock_embedding(),
            persist_directory=vectordb_path
        )
        db.persist()
        # Calculate Page Id's
        chunks_with_ids = self._calculate_chunk_ids(chunks)

        # Add or Update the documents
        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])

        # Only add documents that don't exists in the DB
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)
        
        if len(new_chunks):
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks,ids=new_chunk_ids)
            db.persist()
        else:
            pass
        return SUCCESS
    
    # Calculate chunk id's
    def _calculate_chunk_ids(self, chunks):
        """Calculate chunk id's"""
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get('source')
            page = chunk.metadata.get('page')
            current_page_id = f"{source}:{page}"

            # If the page ID is the same as the last page ID, increament the index
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0
            
            # Calculate the chunk ID
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            # Add it to the page meta-data
            chunk.metadata['id'] = chunk_id
        return chunks
    
    # AWS Bedrock Embedding
    def _aws_bedrock_embedding(self):
        """Perform AWS Bedrock Embedding"""
        aws_bedrock_embedding = BedrockEmbeddings(
            credentials_profile_name="default", region_name="us-east-1"
        )
        return aws_bedrock_embedding
    
    # Query the documents
    def query_documents(self, query: str, k: int = 5) -> List[Document]:
        # Create a Prompt template for Context and Question
        PROMPT_TEMPLATE = """
        Answer based on context: {context}

        Answer the question based on the above context: {question}
        """
        
        """Query the documents"""
        db = Chroma(
            embedding_function=self._aws_bedrock_embedding(),
            persist_directory=self.vectordb_folder
        )
        docs = db.as_retriever().get_relevant_documents(query, k=k)
        return docs