"""Strategist v2 package."""

__all__ = ["StrategistV2"]


def __getattr__(name: str):
	if name == "StrategistV2":
		from .strategist_v2 import StrategistV2  # local import to avoid circular dependency

		return StrategistV2
	raise AttributeError(name)
