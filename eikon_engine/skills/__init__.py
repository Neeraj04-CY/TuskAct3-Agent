"""Skill plugin framework for Strategist V2."""

from .base import SkillBase
from .form_fill import FormFillSkill
from .login import LoginSkill
from .extract import ExtractSkill
from .skill_registry import SkillRegistry

__all__ = [
    "SkillBase",
    "FormFillSkill",
    "LoginSkill",
    "ExtractSkill",
    "SkillRegistry",
]
