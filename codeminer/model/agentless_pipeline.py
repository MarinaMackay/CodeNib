import os
from pathlib import Path
from typing import List, Optional

from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from ..graph.code_graph import CodeGraph
from ..graph.transverse_graph import RepoEntitySearcher, traverse_tree_structure
from ..llm.llm_config import LLMConfig, LLMProvider, create_llm
from ..log_utils import get_logger
from ..scip_interface import SCIPIndexer
from ..types import ROOT_NODE, NodeWithScore

logger = get_logger(__name__)

class Stage1Result(BaseModel):
    """Model for Stage 1 output - file localization."""
    
    files: List[str] = Field(description="List of file paths that need to be edited")


class Stage2Result(BaseModel):
    """Model for Stage 2 output - node localization."""
    
    nodes: List[str] = Field(description="List of node IDs that need to be edited")


class Stage3Result(BaseModel):
    """Model for Stage 3 output - node refinement."""
    
    refined_nodes: List[str] = Field(description="List of refined node IDs that require changes")


class AgentlessPipeline:
    r"""The Agentless approach from the `"Agentless: Demystifying LLM-based
    Software Engineering Agents" <https://arxiv.org/abs/2407.01489>`_ paper.

    :class:`~codeminer.model.AgentlessPipeline` localizes code locations that
    need modification by employing a simplistic three-phase process:

    **Phase 1 - File Localization:** Identifies relevant files using repository
    structure and LLM reasoning:

    .. math::
        \mathcal{F} = \text{LLM}(\text{problem}, \text{structure}(\mathcal{R}))

    where :math:`\mathcal{R}` is the repository and :math:`\mathcal{F}` is the
    set of candidate files.
    #TODO: add dense retrieval to the file localization phase

    **Phase 2 - Node Localization:** Narrows down to specific code entities
    (functions, classes, methods) within the identified files:

    .. math::
        \mathcal{N} = \text{LLM}(\text{problem}, \{\text{structure}(f) \mid f \in \mathcal{F}\})

    where :math:`\mathcal{N}` is the set of candidate nodes (symbols).

    **Phase 3 - Node Refinement:** Validates and refines the final set of nodes
    using full code content:

    .. math::
        \mathcal{N}^* = \text{LLM}(\text{problem}, \{\text{content}(n) \mid n \in \mathcal{N}\})

    where :math:`\mathcal{N}^*` is the refined set of nodes requiring changes.

    The localization results can be used for downstream tasks such as automated
    program repair, code synthesis, or interactive code editing via
    :meth:`~codeminer.model.AgentlessPipeline.query`.

    Args:
        repo_path (str): Path to the repository to analyze.
        repo_commit (str): Git commit hash for version identification.
        llm_model (str, optional): Name of the LLM to use.
            (default: :obj:`"gpt-4o"`)
        llm_provider (str, optional): LLM provider (e.g., "openai", "anthropic").
            (default: :obj:`"openai"`)
        llm_temperature (float, optional): Temperature for LLM sampling.
            (default: :obj:`0.0`)
        llm_max_tokens (int, optional): Maximum tokens for LLM responses.
            (default: :obj:`2048`)
        top_n_files (int, optional): Number of top files to consider in Stage 1.
            (default: :obj:`3`)
        languages (List[str], optional): Programming languages to index.
            If set to :obj:`None`, defaults to :obj:`["python"]`.
            (default: :obj:`None`)
        cache_dir (str, optional): Directory for caching SCIP indices and graphs.
            If set to :obj:`None`, defaults to :obj:`~/.codeminer`.
            (default: :obj:`None`)
    """

    def __init__(
        self,
        # Repo config
        repo_path: str,
        repo_commit: str,
        # LLM configs
        llm_model: str = "gpt-4o",
        llm_provider: str = "openai",
        llm_temperature: float = 0.0,
        llm_max_tokens: int = 2048,
        # Localization configs
        top_n_files: int = 3,
        # Repo processing config
        languages: Optional[List[str]] = None,
        # Cache
        cache_dir: Optional[str] = None,
    ):
        # Attributes
        self.repo_path = os.path.abspath(repo_path)
        self.repo_commit = repo_commit.strip()
        self.top_n_files = top_n_files
        self.languages = languages or ['python']
        self._non_test_file_nodes_cache: Optional[List[str]] = None
        
        # Validate repo
        if not os.path.exists(self.repo_path):
            raise ValueError(f"Repository path does not exist: {self.repo_path}")
        if not os.path.isdir(self.repo_path):
            raise ValueError(f"Repository path is not a directory: {self.repo_path}")
        if not self.repo_commit or not isinstance(self.repo_commit, str):
            raise ValueError("repo_commit must be provided as a non-empty string")
        
        # Initialize cache directory
        cache_dir = Path(cache_dir or Path.home() / ".codeminer")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        repo_name = os.path.basename(self.repo_path)
        commit_identifier = self.repo_commit[:12]
        
        # Initialize CodeGraph via SCIP
        scip_output_dir = cache_dir / f"scip_{repo_name}@{commit_identifier}"
        scip_output_dir.mkdir(parents=True, exist_ok=True)
        
        scip_indexer = SCIPIndexer(self.repo_path, output_dir=str(scip_output_dir))
        self.code_graph: CodeGraph = scip_indexer.run_pipeline(
            project_name=f"{repo_name}@{commit_identifier}",
            skip_level="graph"  # Enable cache: load from graph.pkl if exists
        )
        
        if not self.code_graph:
            raise ValueError("Failed to build code graph")
        
        # Initialize searcher
        self.entity_searcher = RepoEntitySearcher(self.code_graph)
        
        # Initialize LLM
        llm_config = LLMConfig(
            model_name=llm_model,
            provider=LLMProvider(llm_provider),
            max_tokens=llm_max_tokens,
            temperature=llm_temperature,
        )
        self.llm = create_llm(config=llm_config)
        
        # Load prompt templates from files using LangChain
        prompt_dir = Path(__file__).parent / "prompts"
        self.stage1_prompt = PromptTemplate.from_file(
            str(prompt_dir / "agentless_stage1.txt")
        ).template
        self.stage2_prompt = PromptTemplate.from_file(
            str(prompt_dir / "agentless_stage2.txt")
        ).template
        self.stage3_prompt = PromptTemplate.from_file(
            str(prompt_dir / "agentless_stage3.txt")
        ).template
        
        # Create structured LLMs for each stage
        self.structured_llm_stage1 = self.llm.with_structured_output(Stage1Result)
        self.structured_llm_stage2 = self.llm.with_structured_output(Stage2Result)
        self.structured_llm_stage3 = self.llm.with_structured_output(Stage3Result)
        
        logger.info(
            f"AgentlessPipeline initialized: "
            f"repo={repo_name}@{commit_identifier}, "
            f"llm={llm_provider}:{llm_model}, "
            f"top_n_files={top_n_files}, "
            f"cache_dir={cache_dir}"
        )
    
    def query(
        self, 
        problem_statement: str,
    ) -> List[NodeWithScore]:
        files = self._stage_1(problem_statement)
        logger.info(f"Localized {len(files)} files: {files}")
        if not files:
            raise ValueError("No files localized in Stage 1")
        
        nodes = self._stage_2(problem_statement, files)
        logger.info(f"Localized {len(nodes)} nodes: {nodes}")
        if not nodes:
            raise ValueError("No nodes localized in Stage 2")
        
        refined_nodes = self._stage_3(problem_statement, nodes)
        logger.info(f"Localized {len(refined_nodes)} nodes: {refined_nodes}")
        if not refined_nodes:
            raise ValueError("No nodes refined in Stage 3")
        
        return refined_nodes
    
    def _stage_1(self, problem_statement: str) -> List[str]:
        """
        Stage 1: File-level localization using LLM + repository structure.
        """
        # Get repository structure
        structure = traverse_tree_structure(
            code_graph=self.code_graph,
            root=ROOT_NODE,
            direction="downstream",
            hops=3,
            node_type_filter=["file", "directory"],
            edge_type_filter=["contain"]
        )
        
        # Create prompt from loaded template
        prompt = self.stage1_prompt.format(
            problem_statement=problem_statement,
            structure=structure,
            top_n=self.top_n_files
        )
        
        # Query LLM with structured output
        try:
            input_msg = HumanMessage(content=prompt)
            result = self.structured_llm_stage1.invoke([input_msg])
            candidate_files = result.files
            logger.debug(f"[Stage 1] EXTRACTED CANDIDATE FILES: {candidate_files}")
        except Exception as e:
            raise ValueError(f"Error in Stage 1: {e}")

        if not candidate_files:
            raise ValueError("LLM did not return any file candidates")

        # Filter valid file nodes
        files = [
            file for file in candidate_files
            if self.entity_searcher.has_node(file)
        ]

        return files[: self.top_n_files]
    
    def _stage_2(
        self,
        problem_statement: str,
        file_paths: List[str],
    ) -> List[str]:
        """
        Stage 2: Node-level localization using LLM + node structures.
        """
        if not file_paths:
            raise ValueError("Stage 2 invoked with no file candidates")

        # Build file structures
        file_structures = []
        for file_path in file_paths:
            try:
                if self.entity_searcher.has_node(file_path):
                    structure = traverse_tree_structure(
                        code_graph=self.code_graph,
                        root=file_path,
                        direction="downstream",
                        hops=3,
                        node_type_filter=["symbol", "class", "function", "method"],
                        edge_type_filter=["contain", "define"],
                    )
                    if structure:
                        file_structures.append(f"### {file_path} ###\n```\n{structure}\n```\n")
            except Exception as e:
                raise ValueError(f"Error processing file {file_path}: {e}")
        
        if not file_structures:
            raise ValueError("No file structures to process")
        
        # Create prompt from loaded template
        prompt = self.stage2_prompt.format(
            problem_statement=problem_statement,
            file_structures="\n".join(file_structures)
        )
        
        # Query LLM with structured output
        try:
            input_msg = HumanMessage(content=prompt)
            result = self.structured_llm_stage2.invoke([input_msg])
            candidate_nodes = result.nodes
            logger.debug(f"[Stage 2] EXTRACTED CANDIDATE NODES: {candidate_nodes}")
        except Exception as e:
            raise ValueError(f"Error in Stage 2: {e}")

        if not candidate_nodes:
            raise ValueError("LLM did not return any node candidates")

        return candidate_nodes
    
    def _stage_3(
        self,
        problem_statement: str,
        symbols: List[str],
    ) -> List[NodeWithScore]:
        """
        Stage 3: Node-level refinement using full code content.
        """
        if not symbols:
            raise ValueError("Stage 3 invoked with no node candidates")

        code_segments = []
        usable_symbols: List[str] = []

        for symbol_node_id in symbols:
            if not self.entity_searcher.has_node(symbol_node_id):
                raise ValueError(f"Skip missing node: {symbol_node_id}")

            try:
                node_data_list = self.entity_searcher.get_node_data(
                    [symbol_node_id],
                    return_code_content=True,
                    wrap_with_ln=True,
                )
            except Exception as exc:
                raise ValueError(f"Error getting code for {symbol_node_id}: {exc}")

            if not node_data_list:
                raise ValueError(f"No code content available for node: {symbol_node_id}")

            node_data = node_data_list[0]
            code_content = node_data.get("code_content", "")
            if not code_content:
                raise ValueError(f"Empty code content retrieved for node: {symbol_node_id}")

            code_segments.append(
                f"### {symbol_node_id} ###\n```\n{code_content}\n```\n"
            )
            usable_symbols.append(symbol_node_id)

        # Create prompt from loaded template
        prompt = self.stage3_prompt.format(
            problem_statement=problem_statement,
            code_segments="\n".join(code_segments),
        )
        
        # Query LLM with structured output
        try:
            input_msg = HumanMessage(content=prompt)
            result = self.structured_llm_stage3.invoke([input_msg])
            shortlisted_node_ids = result.refined_nodes
            logger.debug(f"[Stage 3] EXTRACTED NODES: {shortlisted_node_ids}")
        except Exception as e:
            raise ValueError(f"Error in Stage 3: {e}")

        candidate_set = set(usable_symbols)
        results: List[NodeWithScore] = []

        for node_id in shortlisted_node_ids:
            if node_id not in candidate_set:
                raise ValueError(f"Discarding node not part of Stage 2 results: {node_id}")
            
            attrs = self.entity_searcher.get_node_data([node_id])[0]
            results.append(NodeWithScore(
                node_name=node_id,
                type=attrs.get("type", "unknown"),
                file=attrs.get("file"),
                start_line=attrs.get("start_line"),
                end_line=attrs.get("end_line"),
                score=0.0,
            ))

        return results
