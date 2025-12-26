import pomdp_py


# Wall indices in clockwise fashion: 0=North, 1=East, 2=South, 3=West
WALL_NAMES = {0: "North", 1: "East", 2: "South", 3: "West"}


class Observation(pomdp_py.Observation):
    """
    Observation in the maze domain consists of:
    - walls: tuple of 4 booleans indicating walls on (North, East, South, West)
    - orientation: string indicating robot's orientation (North, East, South, West)
    - location: rule-based detection of special locations ("Start", "Goal", or None)
    """

    def __init__(self, walls, orientation, location=None):
        """
        Args:
            walls (tuple): 4-tuple of booleans indicating walls on (N, E, S, W)
            orientation (str): robot's orientation ("North", "East", "South", or "West")
            location (str): optional detection of special location ("Start", "Goal", or None)
        """
        if not isinstance(walls, (tuple, list)) or len(walls) != 4:
            raise ValueError("walls must be a tuple/list of 4 booleans")
        if orientation not in ["North", "East", "South", "West"]:
            raise ValueError(f"Invalid orientation: {orientation}")
        
        self.walls = tuple(walls)
        self.orientation = orientation
        self.location = location

    def is_goal(self):
        """Rule-based detection: Goal is when entered via South with walls on N, E, W"""
        # Goal detected when: entered from South + walls on North, East, West
        return (
            self.orientation == "South"
            and self.walls[0]  # North wall
            and self.walls[1]  # East wall
            and not self.walls[2]  # No South wall (entered from there)
            and self.walls[3]  # West wall
        )

    def is_start(self):
        """Rule-based detection: Start is when robot is at starting position
        Based on specific wall configuration at (5, 6) in the example maze"""
        # Start detected by characteristic wall pattern
        # Adjust these rules based on your specific maze layout
        return (
            self.orientation == "North"
            and self.walls[1]  # East wall
            and self.walls[2]  # South wall
            and not self.walls[0]  # No North wall
            and not self.walls[3]  # No West wall
        )

    def __hash__(self):
        return hash((self.walls, self.orientation, self.location))

    def __eq__(self, other):
        if not isinstance(other, Observation):
            return False
        return (
            self.walls == other.walls
            and self.orientation == other.orientation
            and self.location == other.location
        )

    def __str__(self):
        wall_str = "".join(
            [WALL_NAMES[i][0] if self.walls[i] else "-" for i in range(4)]
        )
        return f"Observation({wall_str}, {self.orientation}, {self.location})"

    def __repr__(self):
        return str(self)
