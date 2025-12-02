"""Central location for demo credential access."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class CredentialBundle:
    """Represents username/password pairs for demos."""

    username: str
    password: str


def load_heroku_credentials() -> CredentialBundle:
    """Return credentials for the public HerokuApp demo."""

    username = os.getenv("HEROKUAPP_USERNAME", "tomsmith")
    password = os.getenv("HEROKUAPP_PASSWORD", "SuperSecretPassword!")
    return CredentialBundle(username=username, password=password)
