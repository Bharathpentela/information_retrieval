# Information Retrieval System

## Overview
The Information Retrieval System is designed to efficiently retrieve relevant documents based 
on user queries. Leveraging sentence embeddings and FAISS for similarity search, this system provides
 quick and accurate answers to user queries by analyzing a large corpus of text.

## Features
- **Sentence Embedding**: Utilizes the `all-MiniLM-L6-v2` model for generating dense vector representations of text.
- **FAISS Indexing**: Employs Facebook's AI Similarity Search (FAISS) to perform fast similarity searches.
- **User-Friendly Queries**: Allows users to input natural language queries and retrieve relevant documents.

## Datasets
This project uses the **BeIR** (Benchmarking Information Retrieval) datasets, specifically the **Natural Questions (NQ)** dataset, which includes both a corpus of documents and corresponding queries.

## Installation

### Prerequisites
- Python 3.x
- Pip

### Steps to Install
1.**Clone the repository:**

   https://github.com/Bharathpentela/information_retrieval.git

2.Navigate to the project directory:

    cd information_retrieval



3.Install the required packages
    pip install requests sentence-transformers transformers faiss-cpu
