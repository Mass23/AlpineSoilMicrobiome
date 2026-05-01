"""Importable module for CHELSA BioClim sampling.

Kept separate from src/0b_part1_climate.py because Python modules cannot start
with a digit when imported.

"""

from .0b_part1_climate import run  # noqa: F401
