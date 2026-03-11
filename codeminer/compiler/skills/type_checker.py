"""
Best-effort type checking for skill inputs.

Python prototype — the Rust compiler will replace this with a proper algebraic
type system.  For now we validate required fields and do simple isinstance
checks so that obvious mistakes surface early.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from ...agent.skills.core import SkillMetadata

# Map type-hint strings to Python builtins for simple validation.
_SIMPLE_TYPES: Dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}


@dataclass(slots=True)
class TypeCheckError:
    """A single type-checking diagnostic."""

    skill_id: str
    input_name: str
    expected_type: str
    actual_type: str
    message: str


class TypeChecker:
    """Validate skill inputs against their declared type hints."""

    def check_skill_inputs(
        self,
        metadata: SkillMetadata,
        provided_inputs: Dict[str, Any],
    ) -> List[TypeCheckError]:
        errors: List[TypeCheckError] = []

        for spec in metadata.inputs:
            if spec.required and spec.name not in provided_inputs:
                errors.append(
                    TypeCheckError(
                        skill_id=metadata.skill_id,
                        input_name=spec.name,
                        expected_type=spec.type_hint,
                        actual_type="<missing>",
                        message=f"Required input {spec.name!r} not provided",
                    )
                )
                continue

            if spec.name in provided_inputs:
                value = provided_inputs[spec.name]
                if not self._validate(value, spec.type_hint):
                    errors.append(
                        TypeCheckError(
                            skill_id=metadata.skill_id,
                            input_name=spec.name,
                            expected_type=spec.type_hint,
                            actual_type=type(value).__name__,
                            message=(
                                f"Type mismatch for {spec.name!r}: "
                                f"expected {spec.type_hint}, got {type(value).__name__}"
                            ),
                        )
                    )

        return errors

    # ------------------------------------------------------------------

    @staticmethod
    def _validate(value: Any, type_hint: str) -> bool:
        """Best-effort runtime type check."""
        builtin = _SIMPLE_TYPES.get(type_hint)
        if builtin is not None:
            return isinstance(value, builtin)
        if type_hint.startswith("List["):
            return isinstance(value, list)
        if type_hint.startswith("Dict["):
            return isinstance(value, dict)
        # Anything else: accept (Rust will be strict).
        return True
