"""
Reward model for the maze domain.

Reward structure:
- Goal reached: +10
- Each step: -1
- Hit a wall (invalid move, stay in place): -5
"""

import pomdp_py
from pomdp_py.problems.maze.domain.state import State


class RewardModel(pomdp_py.RewardModel):
    """
    Reward model for the maze domain.
    """

    def __init__(self, maze_map, goal_reward=10, step_penalty=1, wall_penalty=5):
        """
        Args:
            maze_map: MazeMap object
            goal_reward (float): Reward for reaching the goal
            step_penalty (float): Cost for each step
            wall_penalty (float): Penalty for hitting a wall
        """
        self.maze_map = maze_map
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.wall_penalty = wall_penalty

    def sample(self, state, action, next_state):
        """
        Sample a reward from the reward distribution.
        Since rewards are deterministic, just return the computed reward.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            
        Returns:
            float: Reward for this transition
        """
        return self._reward_func(state, action, next_state)

    def _reward_func(self, state, action, next_state):
        """
        Compute the reward for a transition.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            
        Returns:
            float: Reward value
        """
        # Check if goal was reached
        if next_state.position == self.maze_map.goal:
            return self.goal_reward

        # Check if agent hit a wall (position didn't change)
        if next_state.position == state.position:
            return -self.wall_penalty

        # Default: step penalty
        return -self.step_penalty
