"""
Query synthesis agent for SWE-bench instances.

This agent checks out the target repo commit, summarizes the repo, and uses the
Claude code agent to produce natural-language queries at different query types.

Query Types:
- BEHAVIORAL: Pure natural language, no code identifiers
- MODULE_HINT: May mention module/package names
- FILE_HINT: May mention file paths
- SYMBOL_HINT: May mention specific function/class names
- REASONING: Requires reasoning about code relationships
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from claude_agent_sdk import ClaudeAgentOptions, query
from pydantic import BaseModel, Field, ValidationError

from codeminer.dataset.swebench import SwebenchDataset
from codeminer.dataset.utils import (
    CodeLocation,
    GroundTruth,
    QueryType,
    get_prompt_for_query_type,
)
from codeminer.log_utils import get_logger

logger = get_logger(__name__)


class QuerySynthesisResult(BaseModel):
    """Structured output for synthesized queries."""

    question: str = Field(
        description="Single natural-language question describing the issue."
    )
    focus: Optional[str] = Field(
        default=None,
        description="Optional short phrase highlighting the behavior focus.",
    )
    hints: Optional[List[str]] = Field(
        default=None,
        description="Optional progressive hints that could help locate the code.",
    )


class TargetDiscoveryResult(BaseModel):
    """Structured output for repository target discovery."""

    target_files: List[str] = Field(default_factory=list)
    target_symbols: List[str] = Field(default_factory=list)
    rationale: Optional[str] = Field(default=None)


@dataclass
class RepoSnapshot:
    root: Path
    top_level: List[str]
    languages: List[Tuple[str, int]]

    def format_summary(self) -> str:
        parts: List[str] = []
        if self.top_level:
            parts.append("Top-level entries: " + ", ".join(self.top_level))
        if self.languages:
            lang_summary = ", ".join(
                f"{ext} ({count})" for ext, count in self.languages
            )
            parts.append(f"File extensions (top): {lang_summary}")
        return "\n\n".join(parts)


class ClaudeQuerySynthesizer:
    """
    Synthesize natural-language queries using a Claude-backed LLM.

    Supports multiple query types for generating queries with varying
    amounts of code context revealed.
    """

    def __init__(
        self,
        *,
        model: str = "opus",
        max_turns: int = 10,
        allowed_tools: Optional[List[str]] = None,
        permission_mode: str = "bypassPermissions",
        system_prompt: Optional[str] = None,
        query_type: Union[QueryType, str] = QueryType.BEHAVIORAL,
        difficulty_level: Optional[Union[QueryType, str]] = None,
        max_readme_chars: int = 1500,
        max_metadata_chars: int = 800,
        max_top_level: int = 40,
        max_extensions: int = 8,
    ) -> None:
        self.model = model
        self.max_turns = max_turns
        self.allowed_tools = allowed_tools or []
        self.permission_mode = permission_mode
        self.max_readme_chars = max_readme_chars
        self.max_metadata_chars = max_metadata_chars
        self.max_top_level = max_top_level
        self.max_extensions = max_extensions

        # Parse query type (difficulty_level is a deprecated alias).
        if difficulty_level is not None:
            query_type = difficulty_level
        if isinstance(query_type, str):
            self.query_type = QueryType(query_type)
        else:
            self.query_type = query_type

        # Build system prompt based on query type
        base_prompt = (
            "You are a codebase assistant helping create "
            "code search evaluation queries. "
        )
        level_prompt = get_prompt_for_query_type(self.query_type)
        self.system_prompt = system_prompt or (base_prompt + level_prompt)

    def synthesize_query(
        self,
        instance: Dict[str, Any],
        *,
        repo_root: Optional[str] = None,
        cache_dir: Optional[str] = None,
        ground_truth: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Synthesize a query for a single instance.

        Args:
            instance: SWE-bench instance dictionary
            repo_root: Root directory for repository storage
            cache_dir: Cache directory for datasets
            ground_truth: Optional pre-computed ground truth from gt_locate.py

        Returns:
            Dictionary with query and metadata in CodeSearchQuery format
        """
        repo_path = self._checkout_instance(
            instance, repo_root=repo_root, cache_dir=cache_dir
        )
        snapshot = self._snapshot_repo(repo_path)
        discovered = self._discover_targets(instance, snapshot)
        result = self._generate_question(
            instance,
            snapshot,
            ground_truth=ground_truth,
            discovered_targets=discovered,
        )

        instance_id = instance.get("instance_id", "unknown")
        query_id = f"{instance_id}_{self.query_type.value}"

        # Build ground truth structure
        gt = self._build_ground_truth(
            instance,
            ground_truth,
            discovered_targets=discovered,
        )

        return {
            "query_id": query_id,
            "instance_id": instance_id,
            "repo": instance.get("repo"),
            "base_commit": instance.get("base_commit"),
            "query": result["question"],
            "query_type": self.query_type.value,
            "difficulty": self.query_type.value,
            "ground_truth": gt.to_dict() if gt else None,
            "target_files": gt.to_dict()["target_files"] if gt else [],
            "target_symbols": gt.to_dict()["target_symbols"] if gt else [],
            "focus": result.get("focus"),
            "hints": result.get("hints"),
            "repo_snapshot": snapshot.format_summary(),
            "discovered_target_files": discovered.target_files,
            "discovered_target_symbols": discovered.target_symbols,
            "discovery_rationale": discovered.rationale,
        }

    async def synthesize_query_async(
        self,
        instance: Dict[str, Any],
        *,
        repo_root: Optional[str] = None,
        cache_dir: Optional[str] = None,
        ground_truth: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Synthesize a query for a single instance (async version).

        Args:
            instance: SWE-bench instance dictionary
            repo_root: Root directory for repository storage
            cache_dir: Cache directory for datasets
            ground_truth: Optional pre-computed ground truth from gt_locate.py

        Returns:
            Dictionary with query and metadata in CodeSearchQuery format
        """
        repo_path = self._checkout_instance(
            instance, repo_root=repo_root, cache_dir=cache_dir
        )
        snapshot = self._snapshot_repo(repo_path)
        discovered = await self._discover_targets_async(instance, snapshot)
        result = await self._generate_question_async(
            instance,
            snapshot,
            ground_truth=ground_truth,
            discovered_targets=discovered,
        )

        instance_id = instance.get("instance_id", "unknown")
        query_id = f"{instance_id}_{self.query_type.value}"

        # Build ground truth structure
        gt = self._build_ground_truth(
            instance,
            ground_truth,
            discovered_targets=discovered,
        )

        return {
            "query_id": query_id,
            "instance_id": instance_id,
            "repo": instance.get("repo"),
            "base_commit": instance.get("base_commit"),
            "query": result["question"],
            "query_type": self.query_type.value,
            "difficulty": self.query_type.value,
            "ground_truth": gt.to_dict() if gt else None,
            "target_files": gt.to_dict()["target_files"] if gt else [],
            "target_symbols": gt.to_dict()["target_symbols"] if gt else [],
            "focus": result.get("focus"),
            "hints": result.get("hints"),
            "repo_snapshot": snapshot.format_summary(),
            "discovered_target_files": discovered.target_files,
            "discovered_target_symbols": discovered.target_symbols,
            "discovery_rationale": discovered.rationale,
        }

    def synthesize_queries(
        self,
        instances: Iterable[Dict[str, Any]],
        *,
        repo_root: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        for instance in instances:
            try:
                results.append(
                    self.synthesize_query(
                        instance, repo_root=repo_root, cache_dir=cache_dir
                    )
                )
            except Exception as exc:
                instance_id = instance.get("instance_id", "unknown")
                logger.error(
                    "Failed to synthesize query for %s: %s",
                    instance_id,
                    exc,
                    exc_info=True,
                )
                results.append(
                    {
                        "instance_id": instance_id,
                        "repo": instance.get("repo"),
                        "base_commit": instance.get("base_commit"),
                        "error": str(exc),
                    }
                )
        return results

    async def synthesize_queries_async(
        self,
        instances: Iterable[Dict[str, Any]],
        *,
        repo_root: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for instance in instances:
            try:
                results.append(
                    await self.synthesize_query_async(
                        instance, repo_root=repo_root, cache_dir=cache_dir
                    )
                )
            except Exception as exc:
                instance_id = instance.get("instance_id", "unknown")
                logger.error(
                    "Failed to synthesize query for %s: %s",
                    instance_id,
                    exc,
                    exc_info=True,
                )
                results.append(
                    {
                        "instance_id": instance_id,
                        "repo": instance.get("repo"),
                        "base_commit": instance.get("base_commit"),
                        "error": str(exc),
                    }
                )
        return results

    def _checkout_instance(
        self,
        instance: Dict[str, Any],
        *,
        repo_root: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> str:
        dataset = SwebenchDataset(root=cache_dir, repo_root=repo_root, log=False)
        dataset.process_instance(instance, repo_root=repo_root)
        return dataset.get_repo_path(instance, repo_root=repo_root)

    def _snapshot_repo(self, repo_path: str) -> RepoSnapshot:
        root = Path(repo_path)
        top_level = sorted(
            [entry.name for entry in root.iterdir() if not entry.name.startswith(".")]
        )[: self.max_top_level]

        extensions = self._collect_extensions(root)
        return RepoSnapshot(
            root=root,
            top_level=top_level,
            languages=extensions,
        )

    def _collect_extensions(self, root: Path) -> List[Tuple[str, int]]:
        counts: Dict[str, int] = {}
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if self._is_hidden(path):
                continue
            ext = path.suffix.lower() or "no_ext"
            counts[ext] = counts.get(ext, 0) + 1
        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        return sorted_counts[: self.max_extensions]

    def _is_hidden(self, path: Path) -> bool:
        return any(part.startswith(".") for part in path.parts)

    def _build_ground_truth(
        self,
        instance: Dict[str, Any],
        ground_truth: Optional[Dict[str, Any]] = None,
        discovered_targets: Optional[TargetDiscoveryResult] = None,
    ) -> Optional[GroundTruth]:
        """
        Build a GroundTruth object from instance data and/or pre-computed ground truth.

        Args:
            instance: SWE-bench instance dictionary
            ground_truth: Optional pre-computed ground truth from gt_locate.py

        Returns:
            GroundTruth object or None if no ground truth available
        """
        primary_locations: List[CodeLocation] = []

        if ground_truth:
            # Use pre-computed ground truth from gt_locate.py
            target_files = ground_truth.get("target_files", [])
            symbols_modified = ground_truth.get("symbols_modified", [])
            symbols_added = ground_truth.get("symbols_added", [])

            # Add modified symbols as primary locations
            for symbol_id in symbols_modified + symbols_added:
                if ":" in symbol_id:
                    file_path, symbol = symbol_id.split(":", 1)
                    symbol_type = "function" if symbol.endswith("()") else "class"
                    symbol_name = symbol.rstrip("()")
                    primary_locations.append(
                        CodeLocation(
                            file_path=file_path,
                            symbol=symbol_name,
                            symbol_type=symbol_type,
                        )
                    )

            # If no symbols, add file-level locations
            if not primary_locations:
                for file_path in target_files:
                    primary_locations.append(CodeLocation(file_path=file_path))

        if not primary_locations and discovered_targets is not None:
            for file_path in discovered_targets.target_files:
                if file_path:
                    primary_locations.append(CodeLocation(file_path=file_path))
            for symbol_id in discovered_targets.target_symbols:
                if ":" in symbol_id:
                    file_path, symbol = symbol_id.split(":", 1)
                    symbol_name = symbol.rstrip("()")
                    symbol_type = "function" if symbol.endswith("()") else None
                    primary_locations.append(
                        CodeLocation(
                            file_path=file_path,
                            symbol=symbol_name or None,
                            symbol_type=symbol_type,
                        )
                    )

        # Fallback: extract from patch if no ground truth provided
        if not primary_locations and instance.get("patch"):
            from codeminer.dataset.gt_locate import GTLocator

            locator = GTLocator()
            target_files = locator.get_target_files(instance["patch"])
            for file_path in target_files:
                primary_locations.append(CodeLocation(file_path=file_path))

        if not primary_locations:
            return None

        return GroundTruth(primary_locations=primary_locations)

    def _build_constraint_clause(
        self,
        problem_statement: str,
        ground_truth: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build constraint clause based on query type.

        For harder levels (BEHAVIORAL, MODULE_HINT), we restrict what can be mentioned.
        For easier levels (FILE_HINT, SYMBOL_HINT), we allow and even encourage specifics.
        For REASONING level, we require mentioning some symbols but asking about relationships.
        """
        level = self.query_type

        if level == QueryType.BEHAVIORAL:
            avoid_terms = self._extract_avoid_terms(problem_statement)
            if avoid_terms:
                return (
                    "STRICT: You MUST NOT mention any of these identifiers: "
                    + ", ".join(sorted(avoid_terms))
                    + ". Focus purely on observable behavior."
                )
            return (
                "STRICT: Avoid ALL code identifiers, file paths, and technical names."
            )

        elif level == QueryType.MODULE_HINT:
            return (
                "You MAY mention module or package names (e.g., 'the caching module') "
                "but AVOID specific file paths or function/class names."
            )

        elif level == QueryType.FILE_HINT:
            if ground_truth:
                files = ground_truth.get("target_files", [])
                if files:
                    return (
                        f"You SHOULD mention relevant file paths from: {', '.join(files)}. "
                        "But AVOID mentioning specific function or class names."
                    )
            return "You MAY mention specific file paths but AVOID function/class names."

        elif level == QueryType.SYMBOL_HINT:
            if ground_truth:
                symbols = ground_truth.get("symbols_modified", []) + ground_truth.get(
                    "symbols_added", []
                )
                if symbols:
                    return (
                        f"You SHOULD mention relevant symbols from: {', '.join(symbols[:5])}. "
                        "Be specific about which function/class/method is involved."
                    )
            return "You MAY and SHOULD mention specific function/class/method names."

        elif level == QueryType.REASONING:
            if ground_truth:
                symbols = ground_truth.get("symbols_modified", []) + ground_truth.get(
                    "symbols_added", []
                )
                if symbols:
                    return (
                        f"Mention some symbols from: {', '.join(symbols[:3])}. "
                        "But frame the question to require understanding of call chains, "
                        "inheritance, or control flow. Ask 'what calls X', 'which classes "
                        "inherit from Y', or 'how does A interact with B'."
                    )
            return (
                "Frame the question to require reasoning about code relationships "
                "(call chains, inheritance, data flow)."
            )

        return "Focus on the behavior being described."

    def _generate_question(
        self,
        instance: Dict[str, Any],
        snapshot: RepoSnapshot,
        ground_truth: Optional[Dict[str, Any]] = None,
        discovered_targets: Optional[TargetDiscoveryResult] = None,
    ) -> Dict[str, Any]:
        """
        Generate a question at the configured query type.

        Returns a dict with 'question', 'focus', and optionally 'hints'.
        """
        problem_statement = (instance.get("problem_statement") or "").strip()
        synthesis_run_id = instance.get("synthesis_run_id")

        constraint_clause = self._build_constraint_clause(
            problem_statement, ground_truth
        )

        # Build context based on query type
        context_parts = ["Repo summary:\n" + snapshot.format_summary()]

        if synthesis_run_id is not None:
            context_parts.append(
                "Diversity run id: "
                f"{synthesis_run_id}. Produce a distinct query focus from other runs."
            )

        # Add ground truth info for levels that need it
        if ground_truth and self.query_type in (
            QueryType.FILE_HINT,
            QueryType.SYMBOL_HINT,
            QueryType.REASONING,
        ):
            gt_info = []
            if ground_truth.get("target_files"):
                gt_info.append(
                    f"Target files: {', '.join(ground_truth['target_files'])}"
                )
            if ground_truth.get("symbols_modified"):
                gt_info.append(
                    f"Modified symbols: {', '.join(ground_truth['symbols_modified'][:5])}"
                )
            if gt_info:
                context_parts.append(
                    "Ground truth info (use as appropriate):\n" + "\n".join(gt_info)
                )
        elif discovered_targets and self.query_type in (
            QueryType.FILE_HINT,
            QueryType.SYMBOL_HINT,
            QueryType.REASONING,
        ):
            discovered_info = []
            if discovered_targets.target_files:
                discovered_info.append(
                    "Discovered files: "
                    + ", ".join(discovered_targets.target_files[:8])
                )
            if discovered_targets.target_symbols:
                discovered_info.append(
                    "Discovered symbols: "
                    + ", ".join(discovered_targets.target_symbols[:8])
                )
            if discovered_info:
                context_parts.append(
                    "Repository exploration targets:\n" + "\n".join(discovered_info)
                )

        context_parts.append(f"Constraints:\n{constraint_clause}")
        context_parts.append(
            "Output JSON with keys: question (string), focus (string or null), "
            "hints (array of strings or null). Return only JSON."
        )

        user_content = "\n\n".join(context_parts)

        payload = self._run_agent(user_content, cwd=str(snapshot.root))
        payload = self._extract_json_blob(payload)

        try:
            result = QuerySynthesisResult.model_validate_json(payload)
            question = result.question.strip()
            focus = result.focus
            hints = result.hints
        except ValidationError:
            question = self._coerce_question_text(payload)
            focus = None
            hints = None

        # Only sanitize for BEHAVIORAL level
        if self.query_type == QueryType.BEHAVIORAL:
            question = self._sanitize_question(question)

        if not question:
            question = self._fallback_question(discovered_targets)
        if not question.endswith("?"):
            question = question.rstrip(".") + "?"

        return {"question": question, "focus": focus, "hints": hints}

    async def _generate_question_async(
        self,
        instance: Dict[str, Any],
        snapshot: RepoSnapshot,
        ground_truth: Optional[Dict[str, Any]] = None,
        discovered_targets: Optional[TargetDiscoveryResult] = None,
    ) -> Dict[str, Any]:
        """
        Generate a question at the configured query type (async version).

        Returns a dict with 'question', 'focus', and optionally 'hints'.
        """
        problem_statement = (instance.get("problem_statement") or "").strip()
        synthesis_run_id = instance.get("synthesis_run_id")

        constraint_clause = self._build_constraint_clause(
            problem_statement, ground_truth
        )

        # Build context based on query type
        context_parts = ["Repo summary:\n" + snapshot.format_summary()]

        if synthesis_run_id is not None:
            context_parts.append(
                "Diversity run id: "
                f"{synthesis_run_id}. Produce a distinct query focus from other runs."
            )

        # Add ground truth info for levels that need it
        if ground_truth and self.query_type in (
            QueryType.FILE_HINT,
            QueryType.SYMBOL_HINT,
            QueryType.REASONING,
        ):
            gt_info = []
            if ground_truth.get("target_files"):
                gt_info.append(
                    f"Target files: {', '.join(ground_truth['target_files'])}"
                )
            if ground_truth.get("symbols_modified"):
                gt_info.append(
                    f"Modified symbols: {', '.join(ground_truth['symbols_modified'][:5])}"
                )
            if gt_info:
                context_parts.append(
                    "Ground truth info (use as appropriate):\n" + "\n".join(gt_info)
                )
        elif discovered_targets and self.query_type in (
            QueryType.FILE_HINT,
            QueryType.SYMBOL_HINT,
            QueryType.REASONING,
        ):
            discovered_info = []
            if discovered_targets.target_files:
                discovered_info.append(
                    "Discovered files: "
                    + ", ".join(discovered_targets.target_files[:8])
                )
            if discovered_targets.target_symbols:
                discovered_info.append(
                    "Discovered symbols: "
                    + ", ".join(discovered_targets.target_symbols[:8])
                )
            if discovered_info:
                context_parts.append(
                    "Repository exploration targets:\n" + "\n".join(discovered_info)
                )

        context_parts.append(f"Constraints:\n{constraint_clause}")
        context_parts.append(
            "Output JSON with keys: question (string), focus (string or null), "
            "hints (array of strings or null). Return only JSON."
        )

        user_content = "\n\n".join(context_parts)

        payload = await self._run_agent_async(user_content, cwd=str(snapshot.root))
        payload = self._extract_json_blob(payload)

        try:
            result = QuerySynthesisResult.model_validate_json(payload)
            question = result.question.strip()
            focus = result.focus
            hints = result.hints
        except ValidationError:
            question = self._coerce_question_text(payload)
            focus = None
            hints = None

        # Only sanitize for BEHAVIORAL level
        if self.query_type == QueryType.BEHAVIORAL:
            question = self._sanitize_question(question)

        if not question:
            question = self._fallback_question(discovered_targets)
        if not question.endswith("?"):
            question = question.rstrip(".") + "?"

        return {"question": question, "focus": focus, "hints": hints}

    def _discover_targets(
        self,
        instance: Dict[str, Any],
        snapshot: RepoSnapshot,
    ) -> TargetDiscoveryResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._discover_targets_async(instance, snapshot))
        raise RuntimeError(
            "Running inside an active event loop. Use synthesize_query_async instead."
        )

    async def _discover_targets_async(
        self,
        instance: Dict[str, Any],
        snapshot: RepoSnapshot,
    ) -> TargetDiscoveryResult:
        context_parts = ["Repo summary:\n" + snapshot.format_summary()]
        context_parts.append(
            "Explore the repository and identify likely implementation targets.\n"
            "Return strict JSON with keys: target_files (array of relative file paths), "
            "target_symbols (array of `file_path:SymbolName` or `file_path:function_name()`), "
            "rationale (string or null). Keep arrays concise (<= 8)."
        )

        raw_payload = await self._run_agent_async(
            "\n\n".join(context_parts),
            cwd=str(snapshot.root),
        )
        payload = self._extract_json_blob(raw_payload)
        try:
            return TargetDiscoveryResult.model_validate_json(payload)
        except ValidationError:
            heuristic = self._parse_discovery_from_text(raw_payload)
            if heuristic.target_files or heuristic.target_symbols:
                return heuristic
            logger.warning("Target discovery returned non-JSON payload; continuing.")
            return TargetDiscoveryResult()

    def _run_agent(self, prompt: str, *, cwd: Optional[str] = None) -> str:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._run_agent_async(prompt, cwd=cwd))
        raise RuntimeError(
            "Running inside an active event loop. Use synthesize_query_async instead."
        )

    async def _run_agent_async(self, prompt: str, *, cwd: Optional[str] = None) -> str:
        options = ClaudeAgentOptions(
            max_turns=self.max_turns,
            system_prompt=self.system_prompt,
            allowed_tools=self.allowed_tools,
            permission_mode=self.permission_mode,
            model=self.model,
            cwd=cwd,
        )
        text_chunks: List[str] = []
        async for message in query(prompt=prompt, options=options):
            if message.__class__.__name__ != "AssistantMessage":
                continue
            content = getattr(message, "content", None)
            if not content:
                continue
            block_texts: List[str] = []
            for block in content:
                text = getattr(block, "text", None)
                if isinstance(text, str) and text.strip():
                    block_texts.append(text.strip())
            if block_texts:
                text_chunks.append("\n".join(block_texts))
        return "\n".join(text_chunks).strip()

    def _extract_json_blob(self, text: str) -> str:
        text = text.strip()
        if not text:
            return text

        fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text)
        if fenced:
            text = fenced.group(1).strip()

        decoder = json.JSONDecoder()
        for idx, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                _obj, end = decoder.raw_decode(text[idx:])
                return text[idx : idx + end]
            except json.JSONDecodeError:
                continue
        return text

    def _extract_avoid_terms(self, text: str) -> List[str]:
        if not text:
            return []
        file_like = re.findall(
            r"\b[\w/\.-]+\.(?:py|js|ts|rs|go|java|cpp|c|h|hpp)\b", text
        )
        func_like = re.findall(r"\b[A-Za-z_]\w*\(\)", text)
        names = set(file_like + func_like)
        return [name for name in names if len(name) > 2]

    def _sanitize_question(self, text: str) -> str:
        text = re.sub(r"`[^`]+`", "", text)
        text = re.sub(
            r"\b[\w/\.-]+\.(?:py|js|ts|rs|go|java|cpp|c|h|hpp)\b",
            "a module",
            text,
        )
        text = re.sub(r"\b[A-Za-z_]\w*\(\)", "a function", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text

    def _coerce_question_text(self, text: str) -> str:
        if not text:
            return ""
        stripped = text.strip()
        if not stripped:
            return ""
        for line in stripped.splitlines():
            candidate = line.strip().strip("-* ").strip()
            if not candidate:
                continue
            candidate = re.sub(r"^(question|query)\s*:\s*", "", candidate, flags=re.I)
            if len(candidate) >= 12:
                if "?" in candidate:
                    return candidate.split("?", 1)[0].strip() + "?"
                return candidate
        return ""

    def _fallback_question(
        self, discovered_targets: Optional[TargetDiscoveryResult]
    ) -> str:
        first_file = (
            discovered_targets.target_files[0]
            if discovered_targets and discovered_targets.target_files
            else None
        )
        first_symbol = (
            discovered_targets.target_symbols[0]
            if discovered_targets and discovered_targets.target_symbols
            else None
        )
        if self.query_type == QueryType.BEHAVIORAL:
            return (
                "Which code path controls this behavior, and why can an automatic "
                "rewrite change runtime semantics?"
            )
        if self.query_type == QueryType.MODULE_HINT:
            module_hint = (
                first_file.split("/", 1)[0]
                if first_file and "/" in first_file
                else "the relevant module"
            )
            return f"In {module_hint}, where is the logic that governs this behavior?"
        if self.query_type == QueryType.FILE_HINT and first_file:
            return f"What logic in {first_file} is responsible for this behavior?"
        if self.query_type == QueryType.SYMBOL_HINT and first_symbol:
            return f"How does {first_symbol} implement this behavior?"
        if self.query_type == QueryType.REASONING:
            if first_symbol:
                return (
                    "What callers or control-flow paths around "
                    f"{first_symbol} determine this behavior?"
                )
            if first_file:
                return f"What interactions in {first_file} determine this behavior?"
        return "Where in the codebase is the logic responsible for this behavior?"

    def _parse_discovery_from_text(self, text: str) -> TargetDiscoveryResult:
        if not text:
            return TargetDiscoveryResult()
        file_pattern = r"\b[\w./-]+\.(?:py|rs|js|ts|go|java|cpp|c|h|hpp)\b"
        files = list(dict.fromkeys(re.findall(file_pattern, text)))
        symbol_pattern = (
            r"\b[\w./-]+\.(?:py|rs|js|ts|go|java|cpp|c|h|hpp):"
            r"[A-Za-z_][\w.:]*(?:\(\))?\b"
        )
        symbols = list(dict.fromkeys(re.findall(symbol_pattern, text)))
        return TargetDiscoveryResult(
            target_files=files[:8],
            target_symbols=symbols[:8],
            rationale="parsed from non-JSON discovery output",
        )
