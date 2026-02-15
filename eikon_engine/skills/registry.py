from typing import Dict

from .extract import ExtractSkill
from .form_fill import FormFillSkill
from .login import LoginFormSkill
from .extraction.listing_extraction import ListingExtractionSkill

SKILL_REGISTRY: Dict[str, object] = {
    "login_form_skill": LoginFormSkill(),
    "form_fill": FormFillSkill(),
    "extract": ExtractSkill(),
    "listing_extraction_skill": ListingExtractionSkill(),
}


def get_skill(name: str):
    return SKILL_REGISTRY.get(name)
