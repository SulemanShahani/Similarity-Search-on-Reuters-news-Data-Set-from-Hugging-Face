# Document Similarity Search with Sentence Transformers and Faiss

This project provides a tool for embedding text documents using Sentence Transformers and performing similarity search using the Faiss library. Users can input queries to find the most similar documents from a pre-embedded document list.

## Features

- **Embedding with Sentence Transformers**: Utilizes the `sentence-transformers` library to generate embeddings for text documents.
- **Similarity Search with Faiss**: Leverages the Faiss library for efficient similarity search and distance calculations.
- **Interactive Query System**: Allows users to input queries and receive the most similar documents, with an option to continue querying or quit.

## Requirements

- Python 3.7+
- torch
- transformers
- sentence-transformers
- faiss

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/SulemanShahani/document-similarity-search.git
    cd document-similarity-search
    ```

   

2. Install the required packages:

    pip install torch transformers sentence-transformers faiss-cpu
    
## Usage

1. **Prepare Document Embeddings**:
    Ensure you have a list of documents to be embedded. The documents should be stored in `documents_list`.

    ```python
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-MiniLM-L6-v2')
    documents_list = ["Document 1 text", "Document 2 text", ...]

    document_embeddings = model.encode(documents_list, convert_to_tensor=True)
    ```

2. **Set Up Faiss Index**:
    Create a Faiss index for similarity search using the document embeddings.

    ```python
    import faiss

    index = faiss.IndexFlatL2(document_embeddings.shape[1])  # Using L2 (Euclidean) distance
    index.add(document_embeddings.numpy())
    ```

3. **Interactive Query System**:
    Run the following code to start the interactive query system:

    ```python
    while True:
        query = input("Enter your query (or type 'quit' to exit): ")
        if query.lower() == 'quit':
            break

        query_embedding = model.encode([query], convert_to_tensor=True)
        distances, indices = index.search(query_embedding.numpy(), k=5)  # Adjust 'k' as needed

        print("\nTop 5 most similar documents:")
        for i, idx in enumerate(indices[0]):
            print(f"\nDocument {idx+1}:\n{documents_list[idx]}\nDistance: {distances[0][i]}")

    print("Goodbye!")
    ```

## Explanation of Distance Score

The "distance score" represents the dissimilarity between the query vector and the document vectors. It is computed using the Euclidean distance in this example. A lower distance score indicates higher similarity between the query and the document.




