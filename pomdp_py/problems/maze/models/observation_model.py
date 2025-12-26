"""
Observation model for the maze domain.

The robot can observe:
1. Walls on all 4 sides (North, East, South, West)
2. Its own orientation
3. Rule-based detection of special locations (Goal, Start)

The observation is deterministic given the true state and maze layout.
"""

import pomdp_py
from pomdp_py.problems.maze.domain.observation import Observation
from pomdp_py.problems.maze.domain.state import State


class ObservationModel(pomdp_py.ObservationModel):
    """
    Deterministic observation model based on maze layout and robot state.
    """

    def __init__(self, maze_map):
        """
        Args:
            maze_map: MazeMap object containing the maze layout
        """
        self.maze_map = maze_map

    def probability(self, observation, next_state, action):
        """
        Returns P(observation | next_state, action).
        Since the observation is deterministic, returns 1.0 if the observation
        matches the expected observation for the state, 0.0 otherwise.
        
        Args:
            observation: Observation received
            next_state: The state that generated this observation
            action: Action that was taken
            
        Returns:
            float: 1.0 if observation is correct for the state, 0.0 otherwise
        """
        expected_obs = self._get_observation(next_state)
        return 1.0 if observation == expected_obs else 0.0

    def sample(self, next_state, action):
        """
        Sample an observation from the observation distribution.
        Since observations are deterministic, just return the expected observation.
        
        Args:
            next_state: The state that generates the observation
            action: The action that was taken
            
        Returns:
            Observation: The observation for this state
        """
        return self._get_observation(next_state)

    def _get_observation(self, state):
        """
        Get the observation for a given state.
        
        Args:
            state: Robot state
            
        Returns:
            Observation: The observation in this state
        """
        position = state.position
        orientation = state.orientation

        # Check for walls in all 4 directions
        walls = (
            self.maze_map.has_wall(position, "North"),
            self.maze_map.has_wall(position, "East"),
            self.maze_map.has_wall(position, "South"),
            self.maze_map.has_wall(position, "West"),
        )

        # Detect special locations
        location = None
        if position == self.maze_map.goal:
            location = "Goal"
        elif position == self.maze_map.start:
            location = "Start"

        return Observation(walls, orientation, location)

    def get_all_observations(self):
        """
        Get all possible observations in the maze.
        For a maze with W x H cells and 4 orientations, this could be large.
        
        Returns:
            list: List of all possible observations
        """
        # This would need to enumerate all wall configurations and orientations
        # For now, return empty list as it's complex to enumerate
        raise NotImplementedError(
            "get_all_observations not implemented for maze domain"
        )
