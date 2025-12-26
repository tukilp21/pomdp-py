"""
Transition model for the maze domain.

The robot can move in 4 directions (North, East, South, West).
With probability 0.8, the robot moves in the intended direction.
With probability 0.1 each, the robot slips and moves in one of the two
orthogonal directions (perpendicular to the intended direction).
"""

import pomdp_py
import random
from pomdp_py.problems.maze.domain.state import State
from pomdp_py.problems.maze.domain.action import MazeAction


class TransitionModel(pomdp_py.TransitionModel):
    """
    Stochastic transition model with slip probability.
    """

    def __init__(self, slip_prob=0.1, maze_map=None):
        """
        Args:
            slip_prob (float): Probability of slipping to each orthogonal direction
                             Main probability of moving in intended direction: 1 - 2*slip_prob
            maze_map: MazeMap object for checking wall collisions
        """
        self.slip_prob = slip_prob
        self.main_prob = 1.0 - 2 * slip_prob
        self.maze_map = maze_map

    def probability(self, next_state, state, action):
        """
        Returns P(next_state | state, action).
        
        Args:
            next_state: Potential next state
            state: Current state
            action: Action taken
            
        Returns:
            float: Probability of transition
        """
        # Get the possible outcomes (main and two slip directions)
        outcomes = self._get_transition_outcomes(state, action)

        for prob, outcome_state in outcomes:
            if next_state == outcome_state:
                return prob
        return 0.0

    def sample(self, state, action):
        """
        Sample a next state from the transition distribution.
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            State: Next state sampled from the distribution
        """
        outcomes = self._get_transition_outcomes(state, action)

        # Sample based on probabilities
        rand = random.uniform(0, 1)
        cumulative_prob = 0.0
        for prob, next_state in outcomes:
            cumulative_prob += prob
            if rand <= cumulative_prob:
                return next_state

        # Fallback (should not reach here)
        return outcomes[-1][1]

    def _get_transition_outcomes(self, state, action):
        """
        Get all possible transition outcomes and their probabilities.
        
        Returns:
            list: List of (probability, next_state) tuples
        """
        outcomes = []
        action_dir = action.name

        # Get orthogonal directions for slipping
        perpendicular_dirs = self._get_perpendicular_directions(action_dir)

        # Main direction outcome
        next_pos_main = self._move_in_direction(state.position, action_dir)
        next_state_main = State(next_pos_main, state.orientation)
        outcomes.append((self.main_prob, next_state_main))

        # Slip outcomes (two orthogonal directions)
        for slip_dir in perpendicular_dirs:
            next_pos_slip = self._move_in_direction(state.position, slip_dir)
            next_state_slip = State(next_pos_slip, state.orientation)
            outcomes.append((self.slip_prob, next_state_slip))

        return outcomes

    def _get_perpendicular_directions(self, direction):
        """Get the two perpendicular directions to the given direction"""
        perpendicular = {
            "North": ["East", "West"],
            "South": ["East", "West"],
            "East": ["North", "South"],
            "West": ["North", "South"],
        }
        return perpendicular[direction]

    def _move_in_direction(self, position, direction):
        """
        Move one step in the given direction.
        
        Args:
            position (tuple): Current (x, y) position
            direction (str): Direction to move
            
        Returns:
            tuple: New position
        """
        x, y = position
        move_map = {
            "North": (x, y - 1),
            "South": (x, y + 1),
            "East": (x + 1, y),
            "West": (x - 1, y),
        }
        return move_map[direction]

    def get_all_states(self):
        """Get all possible states in the maze (used for value iteration)"""
        # This would need to be implemented based on maze dimensions
        # For now, return empty list as it's not commonly used
        raise NotImplementedError("get_all_states not implemented for maze domain")
