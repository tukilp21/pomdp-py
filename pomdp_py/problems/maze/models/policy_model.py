"""
Policy model for the maze domain.

Provides a uniform random rollout policy for planning algorithms like POMCP.
"""

import pomdp_py
import random
from pomdp_py.problems.maze.domain.action import get_all_actions


class PolicyModel(pomdp_py.RolloutPolicy):
    """
    A simple uniform random policy for the maze domain.
    Used as a rollout policy in simulation-based planners.
    """

    def __init__(self):
        """Initialize the policy model"""
        self._all_actions = get_all_actions()

    def sample(self, state, history=None):
        """
        Sample an action uniformly at random.
        
        Args:
            state: Current state
            history: Action-observation history (optional)
            
        Returns:
            Action: Randomly sampled action
        """
        return random.choice(self._all_actions)

    def rollout(self, state, history=None):
        """
        Perform a rollout (single action sample) from a given state.
        
        Args:
            state: State to rollout from
            history: Action-observation history
            
        Returns:
            Action: Sampled action for rollout
        """
        return self.sample(state, history)

    def get_all_actions(self, state=None, history=None):
        """
        Get all possible actions.
        
        Args:
            state: Current state (optional)
            history: Action-observation history (optional)
            
        Returns:
            list: All possible actions
        """
        return self._all_actions
