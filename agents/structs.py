"""Re-export struct types from dependencies for test and agent use."""

from arc_agi.scorecard import Card, Scorecard
from arcengine.enums import ActionInput, FrameData, GameAction, GameState

__all__ = [
    "ActionInput",
    "Card",
    "FrameData",
    "GameAction",
    "GameState",
    "Scorecard",
]
