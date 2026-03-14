"""
LLM-based skill routing agent.

Given a user query, this agent reads every registered skill's metadata
(skill_id, inputs spec, skill.md documentation) and asks an LLM to decide:
  - which skills to invoke
  - in what order (dependency edges)
  - with what inputs

The output is a list of skill invocation dicts that can be fed directly into
SkillCompiler.compile() without any further modification.

Typical usage:

    from codeminer.agent.routing_agent import SkillRoutingAgent
    from codeminer.llm.litellm_chat import LiteLLMChat

    llm = LiteLLMChat(model="claude-sonnet-4-6")
    agent = SkillRoutingAgent(llm=llm)
    invocations = agent.route(query="find error handling logic")
    # -> [{"skill_id": "embedding_search", "inputs": {...}}, ...]

    result = compiler.compile(invocations)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..llm.litellm_chat import ChatMessage, LiteLLMChat, human_message, system_message
from .skills.core import SkillInputSpec, SkillMetadata
from .skills.registry import SkillRegistry

# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------


class SkillInvocationSchema(BaseModel):
    """A single skill call as decided by the routing LLM."""

    skill_id: str = Field(description="Must match a registered skill_id exactly.")
    inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value pairs for skill inputs. Must include all required inputs.",
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description=(
            "skill_ids whose outputs must be available before this skill runs. "
            "Leave empty for root skills."
        ),
    )


class RoutingPlan(BaseModel):
    """Full routing decision returned by the LLM."""

    reasoning: str = Field(
        description="Brief explanation of why these skills were chosen."
    )
    invocations: List[SkillInvocationSchema] = Field(
        description="Ordered list of skill invocations forming the execution plan."
    )


# ---------------------------------------------------------------------------
# Routing agent
# ---------------------------------------------------------------------------


@dataclass
class SkillRoutingAgent:
    """
    Asks an LLM to select and parameterise skills for a given query.

    The agent:
    1. Formats all registered skills into a prompt (skill_id, inputs, skill.md).
    2. Calls the LLM with structured output (RoutingPlan schema).
    3. Validates the response against the registry.
    4. Returns a list of invocation dicts ready for SkillCompiler.compile().
    """

    llm: LiteLLMChat
    registry: Optional[SkillRegistry] = field(default=None)

    def __post_init__(self) -> None:
        if self.registry is None:
            self.registry = SkillRegistry()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        query: str,
        *,
        task_description: str = "",
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Decide which skills to run for the given query.

        Args:
            query: The user's search query or problem description.
            task_description: Optional extra context about what the caller
                is trying to achieve (e.g. "find the root cause of a bug").
            top_k: Default top_k to inject into inputs when the LLM omits it.

        Returns:
            List of invocation dicts, each with keys "skill_id", "inputs",
            and optionally "depends_on".  Ready for SkillCompiler.compile().

        Raises:
            ValueError: If the LLM response references unknown skill_ids or
                omits required inputs.
        """
        skills = list(self.registry.list_skills().values())
        if not skills:
            raise RuntimeError("SkillRegistry is empty. Load skills before routing.")

        messages = self._build_messages(query, skills, task_description, top_k)
        plan = self._call_llm(messages)
        self._validate(plan, query, top_k)
        return self._to_invocations(plan)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        query: str,
        skills: List[SkillMetadata],
        task_description: str,
        top_k: int,
    ) -> List[ChatMessage]:
        """Build the system + user message pair for the routing call."""
        skill_catalog = self._format_skill_catalog(skills)
        sys_prompt = _SYSTEM_PROMPT_TEMPLATE.format(skill_catalog=skill_catalog)

        user_parts = [f"Query: {query}"]
        if task_description:
            user_parts.append(f"Task context: {task_description}")
        user_parts.append(f"Default top_k: {top_k}")
        user_parts.append(
            "Return a RoutingPlan JSON with the skills that best answer this query."
        )

        return [system_message(sys_prompt), human_message("\n".join(user_parts))]

    def _format_skill_catalog(self, skills: List[SkillMetadata]) -> str:
        """Render all skills as a structured text block for the prompt."""
        blocks: List[str] = []
        for meta in skills:
            inputs_text = _format_inputs(meta.inputs)
            doc = meta.skill_doc.strip() if meta.skill_doc else "(no documentation)"
            block = (
                f"Skill ID: {meta.skill_id}\n"
                f"Type: {meta.skill_type.value}  Cost: {meta.cost.value}\n"
                f"Inputs:\n{inputs_text}\n"
                f"Documentation:\n{doc}"
            )
            blocks.append(block)
        return "\n\n---\n\n".join(blocks)

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, messages: List[ChatMessage]) -> RoutingPlan:
        """Call the LLM and parse the structured response."""
        structured = self.llm.with_structured_output(RoutingPlan)
        return structured.invoke(messages)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, plan: RoutingPlan, query: str, top_k: int) -> None:
        """
        Check the plan for correctness and fill in sensible defaults.

        Mutates plan.invocations in place (adds missing query/top_k inputs).
        Raises ValueError on unrecoverable errors.
        """
        known_ids = set(self.registry.list_skills().keys())
        invoked_ids: set[str] = set()

        for inv in plan.invocations:
            # Unknown skill
            if inv.skill_id not in known_ids:
                raise ValueError(
                    f"LLM returned unknown skill_id {inv.skill_id!r}. "
                    f"Known skills: {sorted(known_ids)}"
                )
            invoked_ids.add(inv.skill_id)

            # Dependency references a skill not in the plan
            for dep in inv.depends_on:
                if dep not in invoked_ids and dep not in {
                    i.skill_id for i in plan.invocations
                }:
                    raise ValueError(
                        f"Skill {inv.skill_id!r} depends on {dep!r} "
                        f"which is not in the routing plan."
                    )

            # Fill in query and top_k if the LLM omitted them
            meta = self.registry.get(inv.skill_id)
            for spec in meta.inputs:
                if spec.name == "query" and "query" not in inv.inputs:
                    inv.inputs["query"] = query
                if spec.name == "top_k" and "top_k" not in inv.inputs:
                    inv.inputs["top_k"] = top_k

            # Check required inputs
            for spec in meta.inputs:
                if spec.required and spec.name not in inv.inputs:
                    raise ValueError(
                        f"Skill {inv.skill_id!r} requires input {spec.name!r} "
                        f"but it was not provided by the LLM."
                    )

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    def _to_invocations(self, plan: RoutingPlan) -> List[Dict[str, Any]]:
        """Convert RoutingPlan to the dict format SkillCompiler expects."""
        result: List[Dict[str, Any]] = []
        for inv in plan.invocations:
            entry: Dict[str, Any] = {
                "skill_id": inv.skill_id,
                "inputs": dict(inv.inputs),
            }
            if inv.depends_on:
                entry["depends_on"] = list(inv.depends_on)
            result.append(entry)
        return result


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are a skill routing agent for a code search system.

Your job is to select the right combination of skills to answer a user query \
about a codebase and produce a valid execution plan.

Rules:
1. Only use skill_ids listed in the catalog below. Never invent new ones.
2. The "inputs" dict must include all required inputs for each skill.
3. The "query" input is always the user's query string verbatim.
4. Use "depends_on" to declare data-flow dependencies between skills \
   (e.g. a rerank skill must depend on the retrieval skill that feeds it).
5. Root skills (no upstream data needed) must have an empty depends_on list.
6. Prefer lower-cost skills unless the query clearly requires semantic understanding.
7. For simple exact-name lookups use bm25_search; \
   for conceptual / intent queries use embedding_search; \
   for maximum coverage use hybrid_search.

Available skills:
{skill_catalog}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_inputs(specs: List[SkillInputSpec]) -> str:
    """Render input specs as a compact indented list."""
    if not specs:
        return "  (none)"
    lines: List[str] = []
    for s in specs:
        req = "required" if s.required else f"optional, default={s.default!r}"
        desc = f" -- {s.description}" if s.description else ""
        lines.append(f"  {s.name} ({s.type_hint}, {req}){desc}")
    return "\n".join(lines)