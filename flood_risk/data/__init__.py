"""Stages 1-3: data sources, collection & integration, preprocessing."""

from .ingestion import load_sources
from .integration import integrate_sources
from .preprocessing import preprocess

__all__ = ["load_sources", "integrate_sources", "preprocess"]
