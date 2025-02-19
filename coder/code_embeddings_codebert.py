"""
Created by Analitika at 18/02/2025
contact@analitika.fr

Improved code for generating CodeBERT embeddings, storing them in FAISS,
and retrieving similar code snippets.
"""

# External imports
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import ast
import faiss
import pickle
import os

# Internal imports
from config import PROJ_ROOT
from coder.directed_graph import analyze_project

DATA_FOLDER = os.path.join(PROJ_ROOT, "coder", "data")

# Load CodeBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
# ".faiss added because of faiss annoying behaviour of adding .faiss
INDEX_NAME = "code_embeddings.index"


def get_code_embedding(code_snippet: str) -> np.ndarray:
    """
    Generate a CodeBERT embedding for a given code snippet.

    Args:
        code_snippet (str): The code snippet to embed.

    Returns:
        np.ndarray: A 1D numpy array representing the embedding.
    """
    # Tokenize the code snippet with truncation and a maximum length of 512 tokens.
    inputs = tokenizer(
        code_snippet, return_tensors="pt", truncation=True, max_length=512
    )

    # Perform inference without gradient tracking.
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the CLS token representation from the last hidden state.
    # Squeeze to remove the batch dimension (shape becomes [hidden_dim,]).
    embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).numpy()
    return embedding


def extract_code_snippets(dependencies: dict) -> tuple:
    """
    Extract code snippets (functions and classes) from files based on dependency info,
    using AST to extract only the relevant code blocks.

    Args:
        dependencies (dict): Dictionary mapping file paths to dependency data.

    Returns:
        tuple: A tuple (code_snippets, metadata) where:
            - code_snippets is a list of code snippet strings.
            - metadata is a list of dicts with keys: 'type', 'name', and 'file'.
    """
    code_snippets = []
    metadata = []

    def get_source_segment(file_path, node):
        """Extracts the exact code block for a function/class from the source file."""
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return "".join(
            lines[node.lineno - 1 : node.end_lineno]
        )  # Extract correct range

    for file, deps in dependencies.items():
        try:
            with open(file, "r", encoding="utf-8") as f:
                source_code = f.read()
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            continue

        try:
            tree = ast.parse(source_code, filename=file)  # Parse the file into an AST
        except SyntaxError as e:
            print(f"Error parsing file {file}: {e}")
            continue

        # Extract functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in deps.get(
                "functions", []
            ):
                snippet = get_source_segment(file, node)
                code_snippets.append(snippet)
                metadata.append({"type": "function", "name": node.name, "file": file})

            elif isinstance(node, ast.ClassDef) and node.name in deps.get(
                "classes", []
            ):
                snippet = get_source_segment(file, node)
                code_snippets.append(snippet)
                metadata.append({"type": "class", "name": node.name, "file": file})

    return code_snippets, metadata


def save_embeddings_and_metadata(
    embeddings: np.ndarray,
    metadata: list,
    index_filename: str = INDEX_NAME,
    metadata_filename: str = "metadata.pkl",
) -> None:
    """
    Save computed embeddings into a FAISS index and store metadata as a pickle file.

    Args:
        embeddings (np.ndarray): Array of embeddings with shape (num_snippets, embedding_dim).
        metadata (list): List of metadata dictionaries corresponding to each embedding.
        index_filename (str): Filename for saving the FAISS index.
        metadata_filename (str): Filename for saving the metadata.
    """
    # Determine the dimensionality of the embeddings.
    d = embeddings.shape[1]
    # Create a FAISS index for L2 (Euclidean) distance.
    index = faiss.IndexFlatL2(d)

    # Add embeddings to the FAISS index.
    index.add(embeddings.astype("float32"))

    # Write the FAISS index to disk.
    faiss.write_index(index, os.path.join(DATA_FOLDER, index_filename))

    # Save metadata to disk using pickle.
    meta_file = os.path.join(DATA_FOLDER, metadata_filename)
    with open(meta_file, "wb") as f:
        pickle.dump(metadata, f)


def search_similar_code(
    query: str,
    top_k: int = 5,
    index_filename: str = INDEX_NAME,
    metadata_filename: str = "metadata.pkl",
) -> list:
    """
    Retrieve similar code snippets for a given query using FAISS.

    Args:
        query (str): Query code snippet.
        top_k (int): Number of similar results to retrieve.
        index_filename (str): Filename of the FAISS index.
        metadata_filename (str): Filename of the metadata pickle.

    Returns:
        list: A list of metadata dictionaries for the similar code snippets.
    """
    # Generate the embedding for the query.
    query_embedding = get_code_embedding(query).astype("float32")

    # Load the FAISS index from disk.
    index = faiss.read_index(os.path.join(DATA_FOLDER, index_filename))

    # Perform the similarity search.
    distances, indices = index.search(np.array([query_embedding]), top_k)

    # Load metadata from disk.
    meta_f = os.path.join(DATA_FOLDER, metadata_filename)
    with open(meta_f, "rb") as f:
        metadata = pickle.load(f)

    # Collect and return results using the indices.
    results = [metadata[i] for i in indices[0]]
    return results


def example():
    # Example usage (for testing purposes)
    # Set the root path of your Python project.
    project_path = r"C:\Users\ecepeda\OneDrive - analitika.fr\Documentos\PROYECTOS\ANALITIKA\PycharmProjects\neural_networks\coder"

    # Step 1: Extract the dependency graph and dependencies from your project.
    dependency_graph, dependencies = analyze_project(project_path)
    print("Dependency extraction complete.")

    # Optional: Visualize the graph (if you wish to review the overall dependency structure).
    # from visualize import visualize_directed_graph  # Import your visualization function
    # visualize_directed_graph(dependency_graph, root_folder=project_path)

    # Step 2: Extract code snippets (functions and classes) using the real project dependencies.
    code_snippets, metadata = extract_code_snippets(dependencies)
    print(f"Extracted {len(code_snippets)} code snippets from the project.")

    # Step 3: Compute embeddings for each code snippet using CodeBERT.
    # (For a large number of snippets, consider batching this step for efficiency.)
    embeddings = np.array([get_code_embedding(snippet) for snippet in code_snippets])
    print("Computed embeddings for all code snippets.")

    # Step 4: Save embeddings and metadata to disk using FAISS and pickle.
    save_embeddings_and_metadata(embeddings, metadata)
    print("Embeddings and metadata saved to disk.")

    # Step 5: Demonstrate retrieval by searching for similar code snippets.
    # Provide a sample query (this should be a real code snippet from your project or similar context).
    query_code = "def fetch_user_data(user_id):"
    similar_snippets = search_similar_code(query_code)

    print("Similar code snippets metadata:")
    for snippet_meta in similar_snippets:
        print(snippet_meta)


if __name__ == "__main__":
    example()
