from __future__ import annotations

from typing import Dict, List, Optional

from .models import Capability

CAPABILITY_REGISTRY: Dict[str, Capability] = {
    "web_navigation": Capability(
        id="web_navigation",
        name="Web Navigation",
        description="Navigate through web pages and follow links",
        skills=[],
        domains=["browser", "navigation"],
        risk_level="low",
    ),
    "credential_entry": Capability(
        id="credential_entry",
        name="Credential Entry",
        description="Fill and submit credential forms when appropriate",
        skills=[],
        domains=["browser", "auth"],
        risk_level="medium",
    ),
    "data_extraction": Capability(
        id="data_extraction",
        name="Data Extraction",
        description="Extract structured data from pages such as listings or details",
        skills=[],
        domains=["browser", "data"],
        risk_level="medium",
    ),
    "artifact_persistence": Capability(
        id="artifact_persistence",
        name="Artifact Persistence",
        description="Persist downloaded or generated files to disk",
        skills=[],
        domains=["browser", "filesystem"],
        risk_level="medium",
    ),
    "auth.login": Capability(
        id="auth.login",
        name="Website Login Automation",
        description="Log into websites by detecting and submitting authentication forms",
        skills=["login_form_skill"],
        domains=["browser", "auth"],
        risk_level="medium",
    ),
    "data.listing_extraction": Capability(
        id="data.listing_extraction",
        name="Listing Data Extraction",
        description="Extract structured data from listing or directory pages",
        skills=["listing_extraction_skill"],
        domains=["browser", "data"],
        risk_level="low",
    ),
}


def get_capability(capability_id: str) -> Optional[Capability]:
    return CAPABILITY_REGISTRY.get(capability_id)


def all_capabilities() -> List[Capability]:
    return list(CAPABILITY_REGISTRY.values())


def capabilities_for_skill(skill_id: str) -> List[Capability]:
    return [cap for cap in CAPABILITY_REGISTRY.values() if skill_id in cap.skills]


__all__ = [
    "CAPABILITY_REGISTRY",
    "get_capability",
    "all_capabilities",
    "capabilities_for_skill",
]
