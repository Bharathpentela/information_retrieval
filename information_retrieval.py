# information_retrieval.py

# Install necessary libraries
!pip install requests
!pip install sentence-transformers
!pip install transformers
!pip install faiss-cpu

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Load datasets
corpus_ds = load_dataset("BeIR/nq", "corpus")
queries_ds = load_dataset("BeIR/nq", "queries")

# Print the first document in the corpus
print("First document in the corpus:", corpus_ds['corpus'][0])

# Print the first query
print("First query:", queries_ds['queries'][0])

# Extracting the corpus and queries
corpus = corpus_ds['corpus']
queries = queries_ds['queries']

# Display the first entry for clarity
print("First entry in the corpus:", corpus[0])
print("First entry in the queries:", queries[0])

# Load a pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for the corpus
corpus_embeddings = model.encode([doc['text'] for doc in corpus], show_progress_bar=True)

# Create embeddings for the queries
query_embeddings = model.encode([query['text'] for query in queries], show_progress_bar=True)

# Save the corpus embeddings
np.save('data/corpus_embeddings.npy', corpus_embeddings)

# Save the query embeddings
np.save('data/query_embeddings.npy', query_embeddings)

# Convert embeddings to numpy array and normalize
corpus_embeddings_np = np.array(corpus_embeddings).astype('float32')
faiss.normalize_L2(corpus_embeddings_np)

# Build a FAISS index
index = faiss.IndexFlatL2(corpus_embeddings_np.shape[1])  # Dimension of embeddings
index.add(corpus_embeddings_np)  # Add corpus embeddings to the index

# Function to retrieve top K similar documents
def retrieve_similar_docs(query, k=5):
    query_embedding = model.encode(query, show_progress_bar=False).astype('float32')
    faiss.normalize_L2(query_embedding.reshape(1, -1))
    
    # Perform the search
    D, I = index.search(query_embedding.reshape(1, -1), k)  # D is distances, I is indices
    return I[0]  # return indices of top K similar documents

# Function to answer the query
def answer_query(query):
    indices = retrieve_similar_docs(query)
    
    # Prepare response from the retrieved documents
    response = []
    for idx in indices:
        response.append(corpus[idx]['text'])  # Assuming 'text' is the field containing the document text
    return response

# Example query
user_query = "What is the capital of France?"
answers = answer_query(user_query)
print("Relevant Answers:")
for answer in answers:
    print(answer)

# Testing with a different query
user_query = "Tell me about the Eiffel Tower."
answers = answer_query(user_query)
print("Relevant Answers:")
for answer in answers:
    print(answer)
