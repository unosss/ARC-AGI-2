"""Reusable deterministic 2D perception and operation solver."""

from .component_programs import (
    LocalCrossRecolor,
    MarkerStamp,
    PeriodicRepair,
    TopologyRecolor,
)
from .operations import OcclusionProgram, ReconstructionPolicy
from .scene import RectangleProposal, Scene2D
from .scene_graph import Component, ComponentScene, PairedSceneGraph, ViewSpec
from .solver import SolveResult, solve, solve_task
from .types import AffineD4, BBox, Candidate, Grid, RelationWitness

__all__ = [
    "AffineD4",
    "BBox",
    "Candidate",
    "Component",
    "ComponentScene",
    "Grid",
    "LocalCrossRecolor",
    "MarkerStamp",
    "OcclusionProgram",
    "PairedSceneGraph",
    "PeriodicRepair",
    "RectangleProposal",
    "ReconstructionPolicy",
    "RelationWitness",
    "Scene2D",
    "SolveResult",
    "TopologyRecolor",
    "ViewSpec",
    "solve",
    "solve_task",
]
