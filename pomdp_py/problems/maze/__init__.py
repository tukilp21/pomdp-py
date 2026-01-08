"""
Maze
====

Problem originally introduced in `Solving POMDPs by Searching the Space of Finite Policies <https://arxiv.org/pdf/1301.6720.pdf>`_

`Quoting from the original paper on problem description`:

    A partially observable stochastic maze: the agent must go from the starting state marked with an "S" to the goal marked with an "G". The problem  is partially observable because the agent cannot perceive its true location, but only its orientation and the presence or the absence of a wall on each side of the square defining its current state.  The problem is stochastic because there is a non-zero probability of slipping, so that the agent does not always know if its last attempt to make a move had any consequence on its actual position in the maze.

.. figure:: https://i.imgur.com/i1RDsrL.png
   :alt: Figure from the paper

   Maze POMDP
"""

from pomdp_py.problems.maze.problem import MazeProblem
from pomdp_py.problems.maze.domain.state import State
from pomdp_py.problems.maze.domain.action import (
    MazeAction,
    MoveNorth,
    MoveSouth,
    MoveEast,
    MoveWest,
    get_all_actions,
)
from pomdp_py.problems.maze.domain.observation import Observation, WALL_NAMES
from pomdp_py.problems.maze.env.env import MazeEnvironment
from pomdp_py.problems.maze.models.transition_model import TransitionModel
from pomdp_py.problems.maze.models.observation_model import ObservationModel
from pomdp_py.problems.maze.models.reward_model import RewardModel
from pomdp_py.problems.maze.models.policy_model import PolicyModel
from pomdp_py.problems.maze.models.components.map import MazeMap

__all__ = [
    "MazeProblem",
    "State",
    "MazeAction",
    "MoveNorth",
    "MoveSouth",
    "MoveEast",
    "MoveWest",
    "get_all_actions",
    "Observation",
    "WALL_NAMES",
    "MazeEnvironment",
    "TransitionModel",
    "ObservationModel",
    "RewardModel",
    "PolicyModel",
    "MazeMap",
]
