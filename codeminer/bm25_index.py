import json
import os
from typing import List, Optional

from llama_index.core import Document
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever

from .graph.code_graph import CodeGraph
from .graph.transverse_graph import wrap_code_snippet
from .types import NODE_TYPE_DIRECTORY, NODE_TYPE_FILE, NodeWithContent, is_symbol_node


class BM25CodeIndexer:
    """
    A class that builds a BM25 index from CodeGraph nodes and provides
    search functionality.
    """

    def __init__(self, code_graph=None, max_k: int = 15, language: str = "english"):
        """
        Initialize the BM25CodeIndexer and optionally build the index immediately.

        Args:
            code_graph: CodeGraph instance containing nodes to index. If provided,
                       the index will be built immediately.
            max_k: Maximum number of results to return in searches
            language: Language for stopword removal and stemming
                      Default is "english" which works well for processing code tokens
                      as it treats special characters as separators
        """
        self.max_k = max_k
        self.language = language
        self.documents = []
        self.nodes = []
        self.retriever = None
        self.code_graph: CodeGraph = None
        self.project_root: Optional[str] = None

        # Build the index immediately if a code_graph is provided
        if code_graph is not None:
            self.build_index_from_graph(code_graph)

    def build_index_from_graph(self, code_graph: CodeGraph) -> BM25Retriever:
        """
        Build a BM25 index from a CodeGraph.

        Args:
            code_graph: CodeGraph instance containing nodes to index
        """
        # Reset the index
        self.documents = []
        self.nodes = []
        self.code_graph = code_graph
        self.project_root = code_graph.project_root

        # Convert graph nodes to documents
        for vertex in code_graph.graph.vs:
            doc = self._convert_vertex_to_document(vertex)
            if doc is not None:
                self.documents.append(doc)
                # Create TextNode from Document for BM25Retriever
                text_node = TextNode(text=doc.text, id_=doc.id_, metadata=doc.metadata)
                self.nodes.append(text_node)

        # Create BM25Retriever
        self.retriever = BM25Retriever.from_defaults(
            nodes=self.nodes, similarity_top_k=self.max_k, language=self.language
        )

        return self.retriever

    def _convert_vertex_to_document(self, vertex):
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

        metadata = {
            "node_id": node_name,
            "type": node_type,
            "name": node_name,
        }

        if node_type == NODE_TYPE_FILE:
            # File node: add file path under "file" to align with graph/search consumers
            metadata["file"] = node_name
            content_lines = [f"File: {node_name}"]
            # Optionally include basic identifiers from file path for tokenization
            content_lines.append(
                f"PathTokens: {node_name.replace('/', ' ').replace('.', ' ')}"
            )
            content = "\n".join(content_lines)
        elif node_type == NODE_TYPE_DIRECTORY:
            # Directory node
            content = f"Directory: {node_name}"
        elif is_symbol_node(node_type):
            # Symbol-like nodes: class/function/method/field/symbol
            file_path = vertex["file"] if "file" in vertex.attributes() else None
            start_line = (
                vertex["start_line"]
                if "start_line" in vertex.attributes()
                and vertex["start_line"] is not None
                else 0
            )
            end_line = (
                vertex["end_line"]
                if "end_line" in vertex.attributes() and vertex["end_line"] is not None
                else 0
            )

            if file_path:
                metadata["file"] = file_path
            # Use simplified symbol name consistent with traverse graph utilities
            simplified_name = (
                node_name.split(":")[-1] if ":" in node_name else node_name
            )
            metadata["name"] = simplified_name
            metadata["start_line"] = start_line
            metadata["end_line"] = end_line

            # Enrich text with name variants to improve BM25 matching
            enriched_name = self._enrich_symbol_name(simplified_name)
            parts = [f"Symbol: {simplified_name}", f"Variants: {enriched_name}"]
            if file_path:
                parts.append(f"File: {file_path}")
            parts.append(f"Lines: {start_line}-{end_line}")
            content = "\n".join(parts)
        else:
            # Unknown or root-like node types: index minimally
            content = f"Node: {node_name} Type: {node_type}"

        # Include additional attributes (except ones we already standardized)
        for key in vertex.attributes():
            if key not in [
                "name",
                "type",
                "file",
                "start_line",
                "end_line",
            ]:
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
            .replace(":", " ")
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

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        return_code_content: bool = False,
        wrap_with_ln: bool = True,
    ) -> List[NodeWithContent]:
        """
        Search the index for nodes matching the query.

        Args:
            query: Search query
            top_k: Number of top results to return (defaults to max_k if not specified)
            return_code_content: Whether to include code content in the results
            wrap_with_ln: Whether to wrap code content with line numbers

        Returns:
            List of NodeWithContent objects containing matched nodes with optional content
        """
        if self.retriever is None:
            raise ValueError(
                "Index has not been built. Call build_index_from_graph first."
            )

        results = self.retriever.retrieve(query)

        # Convert results to NodeWithContent objects
        processed_results = []
        for result_node in results:
            # Extract all metadata directly from the node
            metadata = result_node.node.metadata
            node_name = metadata.get("node_id", "")

            # Get basic node info
            file_path = metadata.get("file")
            start_line = metadata.get("start_line")
            end_line = metadata.get("end_line")
            node_type = metadata.get("type", "unknown")

            # Handle code content if requested
            content = None
            if return_code_content and file_path:
                # Construct full file path using project_root if available
                full_file_path = file_path
                if self.project_root and not os.path.isabs(file_path):
                    full_file_path = os.path.join(self.project_root, file_path)

                try:
                    with open(
                        full_file_path, "r", encoding="utf-8", errors="replace"
                    ) as f:
                        if node_type == NODE_TYPE_FILE:
                            # For file nodes, return entire file content
                            code_content = f.read()
                            if wrap_with_ln:
                                lines = code_content.split("\n")
                                content = wrap_code_snippet(code_content, 1, len(lines))
                            else:
                                content = code_content
                        else:
                            # For symbol nodes, extract specific lines
                            if start_line is not None and end_line is not None:
                                lines = f.readlines()
                                # start_line and end_line are already 0-based and inclusive
                                start_idx = max(0, start_line)
                                end_idx = min(
                                    len(lines), end_line + 1
                                )  # +1 for slice end exclusivity

                                extracted_lines = lines[start_idx:end_idx]

                                # Remove trailing empty lines to avoid extra whitespace in output
                                original_end_idx = len(extracted_lines)
                                while (
                                    extracted_lines
                                    and extracted_lines[-1].strip() == ""
                                ):
                                    extracted_lines.pop()

                                code_content = "".join(extracted_lines)

                                if wrap_with_ln:
                                    # Calculate the actual end line after removing empty lines
                                    lines_removed = original_end_idx - len(
                                        extracted_lines
                                    )
                                    actual_end_line = end_line - lines_removed
                                    # Convert to 1-based for display purposes
                                    content = wrap_code_snippet(
                                        code_content,
                                        start_line + 1,
                                        actual_end_line + 1,
                                    )
                                else:
                                    content = code_content
                except (IOError, UnicodeDecodeError):
                    # If file reading fails, content remains None
                    pass

            # Create NodeWithContent object
            result = NodeWithContent(
                score=float(result_node.score or 0.0),
                node_name=node_name,
                type=node_type,
                file=file_path,
                start_line=start_line,
                end_line=end_line,
                content=content,
            )

            processed_results.append(result)

        # Filter by top_k if specified
        if top_k is not None:
            processed_results = processed_results[:top_k]

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
        # This already persists the corpus and retriever configuration
        self.retriever.persist(directory_path)

        # Save additional metadata including project_root
        metadata = {
            "project_root": (
                str(self.project_root) if self.project_root is not None else None
            ),
            "max_k": self.max_k,
            "language": self.language,
        }
        metadata_file = os.path.join(directory_path, "bm25_metadata.json")
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def load_index(self, directory_path: str):
        """
        Load the index from a directory.

        Args:
            directory_path: Path to load the index from
        """
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory {directory_path} does not exist.")

        # Load the BM25 retriever using its built-in load method
        # This already loads the corpus and configuration
        self.retriever = BM25Retriever.from_persist_dir(directory_path)

        # Load additional metadata including project_root
        metadata_file = os.path.join(directory_path, "bm25_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                self.project_root = metadata.get("project_root")
                self.max_k = metadata.get("max_k", 10)
                self.language = metadata.get("language", "english")
        else:
            # For backward compatibility with indices saved without metadata
            self.project_root = None
