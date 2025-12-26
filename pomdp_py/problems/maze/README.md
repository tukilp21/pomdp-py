# Maze POMDP

A Partially Observable Stochastic Maze navigation problem where the agent must navigate from a starting position to a goal location in a maze, with only partial observability of its state (observes walls and orientation, but not its position).

**Paper Reference**: [Solving POMDPs by Searching the Space of Finite Policies](https://arxiv.org/pdf/1301.6720.pdf)

## Problem Description

### The Task
- Agent must navigate from **Start (S)** to **Goal (G)** in an 11×7 maze
- The agent cannot see its position directly
- The agent can only perceive:
  - Walls in 4 directions (North, South, East, West)
  - Its current orientation
  - Whether it's at the goal or start (detected by wall patterns)
- Motion is stochastic: 80% success, 10% slip in each perpendicular direction

### POMDP Components

#### State Space (S)
- **Position**: (x, y) coordinates in the maze
- **Orientation**: North, East, South, or West
- True state is hidden from the agent

#### Action Space (A)
- **MoveNorth**: Attempt to move north
- **MoveSouth**: Attempt to move south
- **MoveEast**: Attempt to move east
- **MoveWest**: Attempt to move west

#### Observation Space (Z)
- **Walls**: Tuple of 4 booleans (N, S, E, W) indicating wall presence
- **Orientation**: Current facing direction
- **Location**: "Goal", "Start", or None (detected by wall patterns)

Example: `Observation(walls=(-S-), orientation=North, location=None)`

#### Transition Model (T)
- **Slip Probability**: 0.1 (slip to each orthogonal direction)
- **Success Probability**: 0.8 (move in intended direction)
- Non-deterministic movement allows agent to discover walls through slip

```
Move(d):
- 80%: Move in direction d
- 10%: Slip left (perpendicular to d)
- 10%: Slip right (perpendicular to d)
```

#### Observation Model (Z)
- **Deterministic**: Given a state, observation is fully determined
- Walls detected based on maze map
- Orientation always accurate
- Location detected via pattern matching on walls

#### Reward Model (R)
- **Goal Reward**: +10 (reaching the goal)
- **Step Penalty**: -1 (each action taken)
- **Wall Penalty**: -5 (collision with wall)

Example reward sequence:
```
Action 1 (no goal): -1
Action 2 (hit wall): -5
Action 3 (no goal): -1
...
Action N (reach goal): +10
```

### Key Classes

#### MazeProblem
Main POMDP problem class that integrates all components.

```python
from pomdp_py.problems.maze import MazeProblem
import pomdp_py

# Create problem with defaults
maze = MazeProblem.create()

# Or with custom parameters
maze = MazeProblem.create(
    slip_prob=0.1,        # Slip probability (0.1 to each perpendicular)
    goal_reward=20,       # Reward for reaching goal
    step_penalty=2,       # Penalty per step
    wall_penalty=10       # Penalty for hitting wall
)

# Access components
maze.agent                      # The agent
maze.env                        # The environment
maze.maze_map                   # The maze layout
```

#### solve()
Main solving loop with detailed terminal output.

```python
def solve(maze_problem, planner, max_steps=50):
    """
    Run the maze solving loop with detailed terminal output.
    
    Args:
        maze_problem (MazeProblem): Problem instance
        planner: Planning algorithm (POUCT or POMCP)
        max_steps (int): Maximum number of steps
    """
```

### Creating Custom Mazes

#### Using Existing Maze
```python
from pomdp_py.problems.maze import MazeProblem

# Uses the example 11×7 maze
maze = MazeProblem.create()
```

#### Creating Custom Maze
```python
from pomdp_py.problems.maze.models.components.map import MazeMap
from pomdp_py.problems.maze import MazeProblem
from pomdp_py.problems.maze.domain.state import State

# Define custom maze layout
# Walls are defined as line segments in each direction
# Format: {direction: [[(x1,y1), (x2,y2)], ...]}
custom_maze = MazeMap(
    start=(5, 6),
    goal=(5, 2),
    walls={
        "North": [[(0,0), (11,0)], [(2,1), (8,1)]],
        "South": [...],
        "East": [...],
        "West": [...]
    }
)

# Create problem with custom maze
maze = MazeProblem.create(
    maze_map=custom_maze,
    init_state=State(custom_maze.start, "North"),
)
```

## Using Different Planners

### POUCT (Recommended)
Fast, deterministic belief representation.

```python
from pomdp_py.problems.maze import MazeProblem
import pomdp_py

maze = MazeProblem.create()

pouct = pomdp_py.POUCT(
    max_depth=10,
    num_sims=1000,
    exploration_const=50,
    rollout_policy=maze.agent.policy_model,
    show_progress=True
)

solve(maze, pouct, max_steps=50)
```

### POMCP (Particle-Based)
Realistic particle belief representation.

```python
from pomdp_py.problems.maze import MazeProblem
import pomdp_py

maze = MazeProblem.create()

# Convert to particle belief
maze.agent.set_belief(
    pomdp_py.Particles.from_histogram(
        maze.agent.belief, 
        num_particles=100
    ),
    prior=True
)

pomcp = pomdp_py.POMCP(
    max_depth=10,
    num_sims=1000,
    exploration_const=50,
    rollout_policy=maze.agent.policy_model,
    show_progress=True
)

solve(maze, pomcp, max_steps=50)
```

### Value Iteration
Exact planning (if belief is discrete and small).

```python
from pomdp_py.problems.maze import MazeProblem
import pomdp_py

maze = MazeProblem.create()

# Value iteration requires discrete, finite state/observation space
vi = pomdp_py.ValueIteration(
    max_iterations=100,
    discount_factor=0.95
)

solve(maze, vi, max_steps=50)
```

## Customization

### Reward Parameters
```python
maze = MazeProblem.create(
    goal_reward=50,         # Higher goal reward
    step_penalty=0.5,       # Lower step cost
    wall_penalty=20         # Higher wall penalty
)
```

### Initial Belief
```python
import pomdp_py
from pomdp_py.problems.maze import MazeProblem
from pomdp_py.problems.maze.domain.state import State

maze_map = MazeProblem.create().maze_map

# Custom belief: concentrated on start position
init_belief = pomdp_py.Histogram({
    State(maze_map.start, "North"): 1.0
})

maze = MazeProblem.create(init_belief=init_belief)
```

### Planning Parameters
```python
import pomdp_py
from pomdp_py.problems.maze import MazeProblem

maze = MazeProblem.create()

# More aggressive planning
pouct = pomdp_py.POUCT(
    max_depth=20,           # Deeper search tree
    num_sims=5000,          # More simulations
    exploration_const=100,  # More exploration
)

solve(maze, pouct, max_steps=100)
```

## Example: Full Script

```python
#!/usr/bin/env python3

import pomdp_py
from pomdp_py.problems.maze import MazeProblem
from pomdp_py.problems.maze.problem import solve

# Create problem
print("Creating Maze Problem...")
maze = MazeProblem.create(
    slip_prob=0.1,
    goal_reward=10,
    step_penalty=1,
    wall_penalty=5
)

# Setup planner
print("Setting up POUCT planner...")
pouct = pomdp_py.POUCT(
    max_depth=10,
    num_sims=1000,
    exploration_const=50,
    rollout_policy=maze.agent.policy_model,
    show_progress=True
)

# Solve
print("Solving...\n")
solve(maze, pouct, max_steps=50)
```

## Understanding the Output

When running `problem.py`, each step shows:

- **Step N**: Current step number
- **True Position**: Actual robot position (x, y)
- **True Orientation**: Which direction the robot is facing
- **Action**: What the robot tried to do (MoveNorth/South/East/West)
- **Observation**: What the robot perceived (walls + orientation + location)
- **Reward**: Reward for this step (-1, -5, +10, etc.)
- **Cumulative Reward**: Total reward so far
- **Num Sims**: How many simulations the planner ran
- **Planning Time**: How long planning took
- **Belief Particles**: (POMCP only) Number of particles in belief

## Debugging Tips

### Why is the agent not reaching the goal?
1. Check the maze layout and wall definitions
2. Increase `max_depth` and `num_sims` in the planner
3. Verify the observation model is correct

### Why are observations all the same?
1. The start position might be in a location with all walls
2. Check the maze_map wall definitions
3. Try different starting positions

### Performance issues?
1. Reduce `num_sims` for faster but less optimal planning
2. Reduce `max_depth` to limit search horizon
3. Try POUCT instead of POMCP (faster)

## References

- **Paper**: Solving POMDPs by Searching the Space of Finite Policies (https://arxiv.org/pdf/1301.6720.pdf)
- **Framework**: pomdp-py (https://h2r.github.io/pomdp-py/)
- **Example Maze**: 11×7 grid with walls in characteristic pattern

## Dependencies

- pomdp-py (with Cython extensions built)
- numpy
- (Optional) pygame (if using visualization)

## Contact & Notes

This implementation follows the pomdp-py framework conventions and integrates with standard planners (POUCT, POMCP, ValueIteration). The maze problem serves as a good example of:

1. Partial observability (can't see position)
2. Stochastic transitions (slip probability)
3. Deterministic observations
4. Custom POMDP domain definition
