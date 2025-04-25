import os
from typing import Any, Dict, List

from llama_index.core import Document
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever

from .code_graph import CodeGraph


class BM25CodeIndexer:
    """
    A class that builds a BM25 index from CodeGraph nodes and provides
    search functionality.
    """

    def __init__(self, top_k: int = 10, language: str = "english"):
        """
        Initialize the BM25CodeIndexer.

        Args:
            top_k: Number of results to return in searches
            language: Language for stopword removal and stemming
                      Default is "english" which works well for processing code tokens
                      as it treats special characters as separators
        """
        self.top_k = top_k
        self.language = language
        self.documents = []
        self.nodes = []
        self.node_id_to_graph_node = {}
        self.retriever = None

    def build_index_from_graph(self, code_graph: CodeGraph) -> BM25Retriever:
        """
        Build a BM25 index from a CodeGraph.

        Args:
            code_graph: CodeGraph instance containing nodes to index
        """
        # Reset the index
        self.documents = []
        self.nodes = []
        self.node_id_to_graph_node = {}

        # Convert graph nodes to documents
        for vertex in code_graph.graph.vs:
            doc = self._convert_vertex_to_document(vertex, code_graph)
            if doc is not None:
                self.documents.append(doc)
                # Create TextNode from Document for BM25Retriever
                text_node = TextNode(text=doc.text, id_=doc.id_, metadata=doc.metadata)
                self.nodes.append(text_node)
                # Store mapping from document ID to graph node for later retrieval
                self.node_id_to_graph_node[doc.id_] = vertex

        # Create BM25Retriever
        self.retriever = BM25Retriever.from_defaults(
            nodes=self.nodes, similarity_top_k=self.top_k, language=self.language
        )

        return self.retriever

    def _convert_vertex_to_document(self, vertex, code_graph):
        """
        Convert a graph vertex to a Document for indexing.

        Args:
            vertex: Graph vertex to convert
            code_graph: CodeGraph instance

        Returns:
            Document object or None if the vertex couldn't be converted
        """
        node_id = vertex.index
        node_type = vertex["type"] if "type" in vertex.attributes() else "unknown"
        node_name = vertex["name"]

        # Extract content based on node type
        if node_type == "file":
            content = f"File: {node_name}"
            metadata = {"type": "file", "path": node_name}
        elif node_type == "symbol":
            # Get file attribute if exists
            file_path = vertex["file"] if "file" in vertex.attributes() else None

            # Get line range if it exists
            start_line = (
                vertex["start_line"] if "start_line" in vertex.attributes() else None
            )
            end_line = vertex["end_line"] if "end_line" in vertex.attributes() else None

            # Enhance content text with additional variations of the symbol name
            # to improve fuzzy matching and partial matches
            enriched_name = self._enrich_symbol_name(node_name)

            content_parts = [f"Symbol: {node_name}", f"Variants: {enriched_name}"]
            if file_path:
                content_parts.append(f"File: {file_path}")
            if start_line and end_line:
                content_parts.append(f"Lines: {start_line}-{end_line}")

            content = "\n".join(content_parts)

            # Create metadata
            metadata = {"type": "symbol", "name": node_name}
            if file_path:
                metadata["file"] = file_path
            if start_line:
                metadata["start_line"] = start_line
            if end_line:
                metadata["end_line"] = end_line
        else:
            # For unknown node types, use name as content
            content = f"Unknown node: {node_name}"
            metadata = {"type": "unknown"}

        # Add any additional attributes as metadata
        for key in vertex.attributes():
            if key not in ["name", "type", "file", "start_line", "end_line"]:
                metadata[key] = vertex[key]

        # Create a unique ID for the document
        doc_id = f"node_{node_id}"

        return Document(text=content, metadata=metadata, id_=doc_id)

    def _enrich_symbol_name(self, name: str) -> str:
        """
        Create variants of a symbol name to improve fuzzy matching.

        Args:
            name: Original symbol name

        Returns:
            String with multiple variations of the name for improved matching
        """
        enriched = [name]  # Start with the original name

        # Split by common code separators
        parts = []
        for part in (
            name.replace(".", " ")
            .replace("/", " ")
            .replace("_", " ")
            .replace("#", " ")
            .split()
        ):
            if part.strip():
                parts.append(part.strip())

                # Add common prefix-related variants for each part
                if len(part) > 3:  # Only for substantial parts
                    # Add substrings to help with partial matches
                    for i in range(3, len(part) + 1):  # Start with at least 3 chars
                        substring = part[:i]
                        if substring not in parts:
                            parts.append(substring)

        # Add parts to enriched names if not already there
        for part in parts:
            if part not in enriched and len(part) > 1:  # Ignore single characters
                enriched.append(part)

        # For method names, handle common prefix/suffix patterns
        lowercase_name = name.lower()
        if "_" in name:
            # Add versions without underscores
            enriched.append(name.replace("_", ""))

        # Handle common method name prefixes
        prefixes = ["get_", "set_", "is_", "has_"]
        for prefix in prefixes:
            if lowercase_name.startswith(prefix):
                # Add version without the prefix
                without_prefix = name[len(prefix) :]
                if without_prefix not in enriched:
                    enriched.append(without_prefix)
            else:
                # Also try adding prefixes to help match method variants
                with_prefix = prefix + name
                if with_prefix not in enriched:
                    enriched.append(with_prefix)

        return " ".join(enriched)

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the index for nodes matching the query.

        Args:
            query: Search query
            top_k: Optional override for number of results to return

        Returns:
            List of dictionaries containing matched nodes with scores
        """
        if self.retriever is None:
            raise ValueError(
                "Index has not been built. Call build_index_from_graph first."
            )

        results = self.retriever.retrieve(query)

        # Convert results to a more usable format
        processed_results = []
        for result_node in results:
            # Get the original graph node using the document ID
            doc_id = result_node.node.id_
            if doc_id in self.node_id_to_graph_node:
                graph_node = self.node_id_to_graph_node[doc_id]

                # Create result dict with the score from the retriever
                result = {
                    "score": float(
                        result_node.score or 0.0
                    ),  # Ensure score is included and convert to float
                    "node_id": graph_node.index,
                    "node_type": (
                        graph_node["type"]
                        if "type" in graph_node.attributes()
                        else "unknown"
                    ),
                    "name": graph_node["name"],
                }

                # Add all vertex attributes
                for key in graph_node.attributes():
                    if key != "name":  # already included above
                        result[key] = graph_node[key]

                processed_results.append(result)

        return processed_results

    def save_index(self, directory_path: str):
        """
        Save the index to a directory.

        Args:
            directory_path: Path to save the index to
        """
        if self.retriever is None:
            raise ValueError(
                "Index has not been built. Call build_index_from_graph first."
            )

        # Create directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)

        # Save the BM25 retriever using its built-in persist method
        self.retriever.persist(directory_path)

    def load_index(self, directory_path: str):
        """
        Load the index from a directory.

        Args:
            directory_path: Path to load the index from
            code_graph: CodeGraph instance for reference
        """
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory {directory_path} does not exist.")

        # Load the BM25 retriever using its built-in load method
        self.retriever = BM25Retriever.from_persist_dir(directory_path)
