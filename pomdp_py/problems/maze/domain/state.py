"""Defines the State for the maze domain, which is the position of the robot and its orientation.
"""

import pomdp_py
import numpy as np


class State(pomdp_py.State):
    """The state of the maze problem is the robot's position and orientation."""

    ORIENTATIONS = ["North", "East", "South", "West"]  # 0, 1, 2, 3

    def __init__(self, position, orientation):
        """
        Initializes a state in the maze domain.

        Args:
            position (tuple): position of the robot (x, y).
            orientation (str or int): orientation of the robot.
                Can be "North"/"East"/"South"/"West" or 0/1/2/3
        """
        if len(position) != 2:
            raise ValueError("State position must be a tuple of length 2")
        self.position = position
        
        # Normalize orientation
        if isinstance(orientation, str):
            if orientation not in self.ORIENTATIONS:
                raise ValueError(f"Invalid orientation: {orientation}")
            self.orientation = orientation
        elif isinstance(orientation, int):
            if orientation not in [0, 1, 2, 3]:
                raise ValueError(f"Invalid orientation index: {orientation}")
            self.orientation = self.ORIENTATIONS[orientation]
        else:
            raise ValueError("Orientation must be a string or int")

    def __hash__(self):
        return hash((self.position, self.orientation))

    def __eq__(self, other):
        if isinstance(other, State):
            return self.position == other.position and self.orientation == other.orientation
        else:
            return False

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "State(%s, %s)" % (str(self.position), self.orientation)
