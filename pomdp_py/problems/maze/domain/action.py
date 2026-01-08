"""
Navigation actions for the maze domain.
The agent can move North, East, South, or West.
"""

import pomdp_py


class MazeAction(pomdp_py.Action):
    """Base class for maze navigation actions"""

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, MazeAction):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"MazeAction({self.name})"


# Specific navigation actions
class MoveNorth(MazeAction):
    def __init__(self):
        super().__init__("North")


class MoveSouth(MazeAction):
    def __init__(self):
        super().__init__("South")


class MoveEast(MazeAction):
    def __init__(self):
        super().__init__("East")


class MoveWest(MazeAction):
    def __init__(self):
        super().__init__("West")


# Convenience function to get all actions
def get_all_actions():
    """Returns all possible navigation actions"""
    return [MoveNorth(), MoveSouth(), MoveEast(), MoveWest()]
