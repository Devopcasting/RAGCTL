# RAG Command Line Utility

## Overview
The RAG Command Line utility is a tool that enables the embedding of PDF documents and asks questions based on the content of the PDF document. This utility leverages the power of Retrieval Augmented Generation (RAG) technology to provide an interactive way to engage with PDF documents.

## Features
Embed PDF documents for efficient querying and analysis
Ask questions based on the content of the PDF document
Leverage RAG technology for advanced natural language processing capabilities

## Technical Stack
PDF Document Embedding
* AWS Bedrock Embedding: Used for embedding PDF documents into vector representations
Data Storage
* Chroma DB: Used for storing the embedded data
Large Language Model (LLM)
* OLLAMA: Used for generating answers to questions based on the embedded PDF documents

## Getting Started
Prerequisites
* Python 3.10 or later
* pip package manager
* AWS Bedrock Embedding credentials
* Chroma DB
* OLLAMA model

Installation
1. Clone the repository: git clone https://github.com/Devopcasting/RAGCTL
2. Install the required packages: pip install -r requirements.txt
3. Run the utility: python -m ragctl --help