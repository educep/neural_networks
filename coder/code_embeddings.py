"""
Created by Analitika at 18/02/2025
contact@analitika.fr

Improved code for generating CodeBERT embeddings, storing them in FAISS,
and retrieving similar code snippets.
"""

# External imports
from __future__ import annotations
import json
import numpy as np
import networkx as nx
import ast
import faiss
import pickle
import os

# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from openai import OpenAI
from loguru import logger
from pydantic import BaseModel

# Internal imports
from coder.settings import (
    OPENAI_COMPLETIONS_MODEL,
    SMALL_EMBEDDINGS_MODEL,
    DATA_FOLDER,
    OPENAI_API_KEY,
)
from coder.directed_graph import (
    analyze_project,
    visualize_directed_graph_interactive,
    filter_graph,
)
from tools.project_tree import main
from serialize import build_dependencies_files
from coder.prompts.base import prompt_base


class Answer(BaseModel):
    response: str
    code_snippet: str


class CoderAI:
    INDEX_NAME = "code_embeddings.index"
    META_NAME = "metadata.pkl"
    DEPENDENCIES_GRAPH = "dependencies_graph.gml"
    index_path = os.path.join(DATA_FOLDER, INDEX_NAME)
    meta_path = os.path.join(DATA_FOLDER, META_NAME)
    graph_path = os.path.join(DATA_FOLDER, DEPENDENCIES_GRAPH)
    index = None
    metadata = None
    graph = None

    def __init__(self, project_path: str):
        self.project_path = project_path
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        # self.embeddings = OpenAIEmbeddings(
        #     model=OPENAI_EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY
        # )
        # self.llm = ChatOpenAI(
        #     model=OPENAI_COMPLETIONS_MODEL, openai_api_key=OPENAI_API_KEY
        # )

        # self.index_path = os.path.join(DATA_FOLDER, INDEX_NAME)
        self.set_data()

    def set_data(self):
        """
        Set up the data for the project, including generating embeddings,
        storing them in FAISS, and saving metadata.
        """
        if not os.path.exists(self.index_path):
            # Step 1: Extract the dependency graph and dependencies from your project.
            dependency_graph, dependencies = analyze_project(project_path)
            logger.info("Dependency extraction complete.")
            # Save the dependency graph for later use.
            nx.write_gml(dependency_graph, os.path.join(DATA_FOLDER, self.graph_path))

            # Step 2: Extract code snippets (functions and classes) using the real project dependencies.
            code_snippets, metadata = self.extract_code_snippets(dependencies)
            logger.info(
                f"Extracted {len(code_snippets)} code snippets from the project."
            )

            # Step 3: Compute embeddings for each code snippet using CodeBERT.
            # (For a large number of snippets, consider batching this step for efficiency.)
            embeddings = np.array(
                [self.get_code_embedding(snippet) for snippet in code_snippets]
            )
            logger.info("Computed embeddings for all code snippets.")

            # Step 4: Save embeddings and metadata to disk using FAISS and pickle.
            self.save_embeddings_and_metadata(embeddings, metadata)
            logger.info("Embeddings and metadata saved to disk.")

        try:
            # Load index from disk
            self.index = faiss.read_index(self.index_path)
            # Load graph from disk
            self.graph = nx.read_gml(self.graph_path)
            # visualize_directed_graph_interactive(self.graph, self.project_path, True)
            # Load metadata from disk.
            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f)

        except Exception as e:
            logger.error(
                f"Error loading metadata: {e} -> delete all content in {DATA_FOLDER}"
            )
            raise

    def get_code_embedding(self, code_snippet: str) -> np.ndarray:
        """
        Generate a embedding for a given code snippet.

        Args:
            code_snippet (str): The code snippet to embed.

        Returns:
            np.ndarray: A 1D numpy array representing the embedding.
        """
        response = self.client.embeddings.create(
            input=code_snippet, model=SMALL_EMBEDDINGS_MODEL, dimensions=1536
        )
        emb = [data.embedding for data in response.data]
        embedding = np.array(emb)
        norms = np.linalg.norm(embedding)  # Compute L2 norm
        embedding = embedding / norms  # Normalize to unit vectors
        return embedding.astype("float32").squeeze()

    def generate_answer(self, prompt: str) -> Answer | None:
        """
        ai_client: is the client to use: in this project we can use 2: Haskn to treat their Content Library,
                    ANK for the rest
        prompt: is the prompt to send to the model
        temperature=0,  # Controls the randomness in the output generation. The hotter, the more random.
                          A temperature of 1 is a standard setting for creative or varied outputs.
        max_tokens=500, # The maximum length of the model's response.
        top_p=1,        # (or nucleus sampling) this parameter controls the cumulative probability distribution
                          of token selection. A value of 1 means no truncation, allowing all tokens to be considered
                          for selection based on their probability.
        frequency_penalty=0,  # Adjusts the likelihood of the model repeating the same line verbatim.
                                Setting this to 0 means there's no penalty for frequency, allowing the model to freely
                                repeat information as needed.
        presence_penalty=0,  # Alters the likelihood of introducing new concepts into the text.
                               A penalty of 0 implies no adjustment, meaning the model is neutral about introducing
                               new topics or concepts.
        from ai_prompts import CleanContent
        aa =  generate_answer("say hello", CleanContent)
        """

        try:
            completion = self.client.beta.chat.completions.parse(
                model=OPENAI_COMPLETIONS_MODEL,
                temperature=0,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                response_format=Answer,
            )

            return completion.choices[0].message.parsed
        except KeyError as e:
            # Handle any issues with missing keys in the response
            logger.error(f"KeyError: {e} - The expected key is not in the response.")
            logger.info(
                "An error occurred: the response structure was not as expected."
            )
            return

        except Exception as e:
            # Handle any other general exceptions (e.g., network errors, API issues)
            logger.error(f"An error occurred: {e}")
            logger.info("An error occurred while generating the response.")
            return

    def extract_code_snippets(self, dependencies: dict) -> tuple:
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
                tree = ast.parse(
                    source_code, filename=file
                )  # Parse the file into an AST
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
                    metadata.append(
                        {"type": "function", "name": node.name, "file": file}
                    )

                elif isinstance(node, ast.ClassDef) and node.name in deps.get(
                    "classes", []
                ):
                    snippet = get_source_segment(file, node)
                    code_snippets.append(snippet)
                    metadata.append({"type": "class", "name": node.name, "file": file})

        return code_snippets, metadata

    def save_embeddings_and_metadata(
        self, embeddings: np.ndarray, metadata: list
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
        # index = faiss.IndexFlatL2(d)
        # Create a FAISS index for Inner product.
        index = faiss.IndexFlatIP(d)  # Inner product index

        # Add embeddings to the FAISS index.
        index.add(embeddings.astype("float32"))

        # Write the FAISS index to disk.
        faiss.write_index(index, self.index_path)

        # Save metadata to disk using pickle.
        with open(self.meta_path, "wb") as f:
            pickle.dump(metadata, f)

    def search_similar_code(self, query: str, top_k: int = 5) -> list:
        """
        Retrieve similar code snippets for a given query using FAISS.

        Args:
            query (str): Query code snippet.
            top_k (int): Number of similar results to retrieve.

        Returns:
            list: A list of metadata dictionaries for the similar code snippets.
        """
        # Generate the embedding for the query.
        query_embedding = self.get_code_embedding(query)

        # Perform the similarity search.
        cos_simil, indices = self.index.search(np.array([query_embedding]), top_k)

        for dist, idx in zip(cos_simil[0], indices[0]):
            print(f"Index: {idx}, Distance: {dist}")

        # Collect and return results using the indices.
        results = [self.metadata[i] for i in indices[0]]
        return results

    def get_dependencies_subgraph(self, file_paths: list, level: int = 5) -> nx.Graph:
        """
        Retrieve a subgraph of dependencies for the given file paths up to a specified level (radius).

        Args:
            graph (nx.Graph): The full dependency graph.
            file_paths (list): List of file paths (strings) extracted from metadata.
            level (int): The maximum number of hops (dependency levels) from nodes associated with the file.

        Returns:
            nx.Graph: A subgraph containing all nodes reachable within 'level' hops from any node associated with one of the files.
        """
        selected_nodes = set()
        # removes unknowns
        filtered_graph = filter_graph(self.graph, "remove", "unknown")
        # Nodes that belong to a file include the file path in their label (e.g., "C:/path/to/file.py::function_name")
        for node in filtered_graph.nodes():
            for file in file_paths:
                if file in node:
                    selected_nodes.add(node)
                    break

        # For each selected node, get its ego graph up to the specified radius
        sub_nodes = set()
        for node in selected_nodes:
            ego = nx.ego_graph(filtered_graph, node, radius=level)
            sub_nodes.update(ego.nodes())

        # Create and return the subgraph
        subgraph = filtered_graph.subgraph(sub_nodes).copy()
        return subgraph


if __name__ == "__main__":
    project_path = r"C:\Users\ecepeda\OneDrive - analitika.fr\Documentos\PROYECTOS\ANALITIKA\PycharmProjects\neural_networks\coder"

    coder = CoderAI(project_path)

    # Provide a sample query (this should be a real code snippet from your project or similar context).
    query_code = "where is the CoderAI class"
    similar_snippets = coder.search_similar_code(query_code)

    # Extract unique file paths from the returned metadata.
    file_paths = list({snippet_meta["file"] for snippet_meta in similar_snippets})
    # Retrieve a dependency subgraph with dependencies up to 3 levels for these files.
    dep_subgraph = coder.get_dependencies_subgraph(file_paths, level=3)
    visualize_directed_graph_interactive(dep_subgraph, coder.project_path)

    print("Dependencies:")
    dependencies, dep_files = [], []
    for u, v, data in dep_subgraph.edges(data=True):
        u_file = u.split("::")[0]
        v_file = v.split("::")[0]
        if u_file not in dep_files:
            dep_files.append(u_file)
        if v_file not in dep_files:
            dep_files.append(v_file)
        msg = f"{u.replace(project_path, '.')} -> {v.replace(project_path, '.')}: {data['type']}"
        if u not in dependencies:
            dependencies.append(msg)

    file_content = build_dependencies_files(dep_files)
    project_files = json.dumps(file_content)
    # Visualize the subgraph using your interactive visualization function.
    project_structure = main(project_path, only_py=True, write_file=False)

    project_description = """
        This is a project develops a RAG for helping developers to improve coding in a
        more effective and faster way. We use OpenAI to compute embeddings and retrieve the
        snippets and files more relevant to the query. The project is developed in Python.
        """

    task_question = "How do I implement a chatbot in Streamlit using the CoderAI class"

    prompt = prompt_base.format(
        project_name="Code IA-RAG",
        project_description=project_description,
        project_files=project_files,
        dependencies_files=dependencies,
        project_structure=project_structure,
        task_question=task_question,
    )

    ans_ = coder.generate_answer(prompt)
    print(ans_.response)
    print(ans_.code_snippet)
