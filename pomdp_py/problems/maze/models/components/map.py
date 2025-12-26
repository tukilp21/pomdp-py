import numpy as np


class MazeMap:
    """
    Represents the maze layout with walls defined as line segments in each direction.
    This allows for flexible and complex maze structures.
    """

    def __init__(self, start, goal, walls):
        """
        Args:
            start (tuple): (x, y) position of the start location
            goal (tuple): (x, y) position of the goal location
            walls (dict): Dictionary with keys "North", "East", "South", "West"
                         Each value is a list of line segments: [(x1, y1), (x2, y2)]
                         representing walls in that direction
        """
        self.start = start
        self.goal = goal
        self.walls = walls
        self._validate_walls()

    def _validate_walls(self):
        """Validate that walls dictionary has correct structure"""
        valid_dirs = {"North", "East", "South", "West"}
        if not all(d in valid_dirs for d in self.walls.keys()):
            raise ValueError(f"Wall directions must be in {valid_dirs}")

    def has_wall(self, position, direction):
        """
        Check if there's a wall at a given position in a given direction.
        
        Args:
            position (tuple): (x, y) position to check
            direction (str): "North", "East", "South", or "West"
            
        Returns:
            bool: True if there's a wall in that direction
        """
        if direction not in self.walls:
            return False

        x, y = position
        for wall_segment in self.walls.get(direction, []):
            # Check if position is on this wall segment
            if self._is_on_segment(position, wall_segment, direction):
                return True
        return False

    def _is_on_segment(self, position, segment, direction):
        """Check if a position is on a given wall segment"""
        x, y = position
        (x1, y1), (x2, y2) = segment

        # Normalize segment so min comes first
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)

        # Check if position is within segment bounds
        if min_x <= x <= max_x and min_y <= y <= max_y:
            return True
        return False

    def get_valid_actions(self, position, orientation):
        """
        Get valid actions from a given position (actions that don't hit walls).
        
        Args:
            position (tuple): (x, y) position
            orientation (str): current orientation
            
        Returns:
            list: List of valid action directions
        """
        valid_actions = []
        for direction in ["North", "East", "South", "West"]:
            if not self.has_wall(position, direction):
                valid_actions.append(direction)
        return valid_actions

    @staticmethod
    def create_example_maze():
        """
        Creates the example 11x7 maze from the paper.
        Maze structure from "Solving POMDPs by Searching the Space of Finite Policies"
        
        Grid is 11 wide (0-10) x 7 tall (0-6)
        Start at (5, 6), Goal at (5, 2)
        """
        start = (5, 6)
        goal = (5, 2)

        # Define walls as cells that have walls on specific sides
        # For each cell, we define which directions have walls
        walls = {
            "North": [
                # Top boundary row (y=0)
                [(0, 0), (10, 0)],
                # Interior walls
                [(0, 1), (1, 1)],
                [(3, 1), (8, 1)],
                [(0, 2), (3, 2)],
                [(6, 2), (8, 2)],
                [(10, 2), (10, 2)],
                [(0, 3), (2, 3)],
                [(4, 3), (5, 3)],
                [(8, 3), (8, 3)],
                [(0, 4), (0, 4)],
                [(3, 4), (5, 4)],
                [(8, 4), (10, 4)],
                [(1, 5), (2, 5)],
                [(4, 5), (5, 5)],
                [(8, 5), (8, 5)],
            ],
            "South": [
                # Bottom boundary row (y=7)
                [(0, 7), (10, 7)],
                # Interior walls (same structure mirrored)
                [(0, 2), (1, 2)],
                [(3, 2), (8, 2)],
                [(0, 3), (3, 3)],
                [(6, 3), (8, 3)],
                [(10, 3), (10, 3)],
                [(0, 4), (2, 4)],
                [(4, 4), (5, 4)],
                [(8, 4), (8, 4)],
                [(0, 5), (0, 5)],
                [(3, 5), (5, 5)],
                [(8, 5), (10, 5)],
                [(1, 6), (2, 6)],
                [(4, 6), (5, 6)],
                [(8, 6), (8, 6)],
            ],
            "East": [
                # Right boundary column (x=11)
                [(11, 0), (11, 7)],
                # Interior walls
                [(1, 1), (1, 2)],
                [(3, 1), (3, 1)],
                [(5, 1), (5, 2)],
                [(6, 1), (6, 1)],
                [(9, 1), (9, 2)],
                [(10, 1), (10, 2)],
                [(2, 3), (2, 3)],
                [(4, 3), (4, 4)],
                [(7, 3), (7, 3)],
                [(9, 3), (9, 3)],
                [(1, 4), (1, 4)],
                [(3, 4), (3, 5)],
                [(6, 4), (6, 4)],
                [(9, 5), (9, 6)],
                [(10, 5), (10, 5)],
                [(2, 6), (2, 6)],
                [(5, 6), (5, 6)],
            ],
            "West": [
                # Left boundary column (x=0)
                [(0, 0), (0, 7)],
                # Interior walls
                [(2, 1), (2, 2)],
                [(4, 1), (4, 1)],
                [(6, 1), (6, 2)],
                [(7, 1), (7, 1)],
                [(10, 1), (10, 2)],
                [(1, 1), (1, 1)],
                [(3, 3), (3, 3)],
                [(5, 3), (5, 4)],
                [(8, 3), (8, 3)],
                [(10, 3), (10, 3)],
                [(2, 4), (2, 4)],
                [(4, 4), (4, 5)],
                [(7, 4), (7, 4)],
                [(10, 5), (10, 6)],
                [(1, 5), (1, 5)],
                [(3, 6), (3, 6)],
                [(6, 6), (6, 6)],
            ],
        }

        return MazeMap(start, goal, walls)

    def visualize_simple(self):
        """
        Simple visualization that just shows start and goal on a grid.
        Useful for debugging the maze structure.
        """
        max_x = 11
        max_y = 7
        
        grid = [['.' for _ in range(max_x)] for _ in range(max_y)]
        
        # Mark start and goal
        sx, sy = self.start
        gx, gy = self.goal
        if 0 <= sy < max_y and 0 <= sx < max_x:
            grid[sy][sx] = 'S'
        if 0 <= gy < max_y and 0 <= gx < max_x:
            grid[gy][gx] = 'G'
        
        # Print with row numbers
        print("  0123456789A")
        for y, row in enumerate(grid):
            print(f"{y} {''.join(row)}")
    
    def visualize_walls_debug(self):
        """
        Debug visualization showing walls for each cell.
        Format: NSEW (North, South, East, West) where - means wall, . means open
        """
        max_x = 11
        max_y = 7
        
        print("\nWall Debug View (NSEW format: - = wall, . = open)")
        print("  ", end="")
        for x in range(max_x):
            print(f" {x:2}", end="")
        print()
        
        for y in range(max_y):
            print(f"{y} ", end="")
            for x in range(max_x):
                n = '-' if self.has_wall((x, y), "North") else '.'
                s = '-' if self.has_wall((x, y), "South") else '.'
                e = '-' if self.has_wall((x, y), "East") else '.'
                w = '-' if self.has_wall((x, y), "West") else '.'
                print(f"{n}{s}{e}{w}", end="")
            print()
    
    def visualize(self):
        """
        Print the maze using a grid with walls shown between cells.
        Each cell is shown as a 3x3 block in ASCII.
        """
        max_x = 11
        max_y = 7
        
        # Create larger grid to show walls between cells
        # Each cell needs 2 chars (cell + wall), plus 1 for final edge
        display = []
        
        for y in range(max_y):
            # Row for cell content and horizontal walls below
            cell_row = []
            wall_row = []
            
            for x in range(max_x):
                # Check walls
                has_north = self.has_wall((x, y), "North")
                has_south = self.has_wall((x, y), "South")
                has_west = self.has_wall((x, y), "West")
                has_east = self.has_wall((x, y), "East")
                
                # Cell content
                if (x, y) == self.start:
                    cell_char = 'S'
                elif (x, y) == self.goal:
                    cell_char = 'G'
                else:
                    cell_char = '.'
                
                # Left wall or space
                if has_west:
                    cell_row.append('|')
                else:
                    cell_row.append(' ')
                
                # Cell
                cell_row.append(cell_char)
                
                # Bottom wall of this cell
                if has_south:
                    wall_row.append('-')
                else:
                    wall_row.append(' ')
                wall_row.append('-' if has_south else ' ')
            
            # Right edge
            has_east_edge = self.has_wall((max_x-1, y), "East")
            cell_row.append('|' if has_east_edge else ' ')
            
            display.append(''.join(cell_row))
            if y < max_y - 1 or y == max_y - 1:  # Show bottom wall for last row
                display.append(''.join(wall_row) + ' ')
        
        # Print
        for line in display:
            print(line)


# Example maze dict (for reference)
example_mapdict = {
    "start": (5, 6),
    "goal": (5, 2),
    "walls": {
        "North": [
            [(0, 0), (11, 0)],
            [(2, 1), (8, 1)],
            [(4, 2), (6, 2)],
        ],
        "South": [
            [(0, 7), (11, 7)],
            [(1, 6), (9, 6)],
            [(3, 5), (7, 5)],
            [(5, 3), (5, 3)],
        ],
        "West": [
            [(0, 0), (0, 7)],
            [(1, 1), (5, 5)],
            [(2, 1), (2, 4)],
            [(3, 2), (3, 4)],
            [(4, 2), (4, 3)],
            [(5, 3), (5, 3)],
        ],
        "East": [
            [(5, 3), (5, 3)],
            [(6, 2), (6, 3)],
            [(7, 2), (7, 4)],
            [(8, 1), (8, 4)],
            [(9, 1), (9, 5)],
            [(10, 0), (10, 6)],
        ],
    },
}


def main():
    """Visualize the example maze"""
    print("Maze Visualization")
    print("=" * 40)
    print("Legend: . = free space, - = wall, S = start, G = goal")
    print()
    
    maze = MazeMap.create_example_maze()
    maze.visualize()


if __name__ == "__main__":
    main()
