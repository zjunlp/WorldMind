"""
WorldMind Module for EmbodiedBench

This package provides the WorldMind components for embodied AI environments,
including state prediction, discrimination, reflection, and knowledge management.

Supported environments:
- Habitat: Home environment with pick/place/navigation tasks
- Alfred: Kitchen environment with cooking and household tasks
- Navigation: Indoor navigation tasks
"""

from embodiedbench.worldmind import habitat
from embodiedbench.worldmind import alfred
from embodiedbench.worldmind import navigation

__all__ = ['habitat', 'alfred', 'navigation']
