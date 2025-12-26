"""
The Maze POMDP Problem.

Problem originally introduced in `Solving POMDPs by Searching the Space of Finite Policies
<https://arxiv.org/pdf/1301.6720.pdf>`_

A partially observable stochastic maze: the agent must go from the starting state
marked with an "S" to the goal marked with a "G". The problem is partially observable
because the agent cannot perceive its true location, but only its orientation and the
presence or absence of a wall on each side of the square defining its current state.
The problem is stochastic because there is a non-zero probability of slipping, so that
the agent does not always know if its last attempt to make a move had any consequence
on its actual position in the maze.
"""

import pomdp_py
from pomdp_py.problems.maze.domain.state import State
from pomdp_py.problems.maze.domain.action import get_all_actions, MazeAction
from pomdp_py.problems.maze.domain.observation import Observation
from pomdp_py.problems.maze.env.env import MazeEnvironment
from pomdp_py.problems.maze.models.transition_model import TransitionModel
from pomdp_py.problems.maze.models.observation_model import ObservationModel
from pomdp_py.problems.maze.models.reward_model import RewardModel
from pomdp_py.problems.maze.models.policy_model import PolicyModel
from pomdp_py.problems.maze.models.components.map import MazeMap
import random


class MazeProblem(pomdp_py.POMDP):
    """
    A MazeProblem is instantiated with a maze layout and initial conditions.
    It contains the agent and environment, along with all their models.
    """

    def __init__(
        self,
        maze_map,
        init_state,
        init_belief,
        slip_prob=0.1,
        goal_reward=10,
        step_penalty=1,
        wall_penalty=5,
    ):
        """
        Args:
            maze_map (MazeMap): The maze layout
            init_state (State): Initial true state of the robot
            init_belief (pomdp_py.Distribution): Initial belief of the agent
            slip_prob (float): Probability of slipping to each orthogonal direction
            goal_reward (float): Reward for reaching the goal
            step_penalty (float): Cost for each step
            wall_penalty (float): Penalty for hitting a wall
        """
        # Create models
        transition_model = TransitionModel(slip_prob=slip_prob, maze_map=maze_map)
        observation_model = ObservationModel(maze_map)
        reward_model = RewardModel(
            maze_map,
            goal_reward=goal_reward,
            step_penalty=step_penalty,
            wall_penalty=wall_penalty,
        )
        policy_model = PolicyModel()

        # Create agent
        agent = pomdp_py.Agent(
            init_belief,
            policy_model,
            transition_model,
            observation_model,
            reward_model,
        )

        # Create environment
        env = MazeEnvironment(init_state, transition_model, reward_model, maze_map)

        # Initialize POMDP
        super().__init__(agent, env, name="Maze")

        self.maze_map = maze_map

    @staticmethod
    def create(
        maze_map=None,
        init_state=None,
        init_belief=None,
        slip_prob=0.1,
        goal_reward=10,
        step_penalty=1,
        wall_penalty=5,
    ):
        """
        Factory method to create a MazeProblem with default or custom parameters.
        
        Args:
            maze_map (MazeMap): Maze layout (default: example maze)
            init_state (State): Initial true state (default: maze start position)
            init_belief (Distribution): Initial belief (default: uniform over reachable states)
            slip_prob (float): Slip probability
            goal_reward (float): Goal reward
            step_penalty (float): Step penalty
            wall_penalty (float): Wall penalty
            
        Returns:
            MazeProblem: Configured maze problem instance
        """
        # Use example maze if not provided
        if maze_map is None:
            maze_map = MazeMap.create_example_maze()

        # Use maze start position if not provided
        if init_state is None:
            init_state = State(maze_map.start, "North")

        # Create uniform belief over some reasonable set of states
        if init_belief is None:
            # Simple uniform belief over a few possible starting positions
            init_belief = pomdp_py.Histogram(
                {State(maze_map.start, orientation): 0.25 for orientation in ["North", "East", "South", "West"]}
            )

        return MazeProblem(
            maze_map,
            init_state,
            init_belief,
            slip_prob=slip_prob,
            goal_reward=goal_reward,
            step_penalty=step_penalty,
            wall_penalty=wall_penalty,
        )


def solve(maze_problem, planner, max_steps=50):
    """
    Run the maze solving loop with detailed terminal output.
    
    Args:
        maze_problem (MazeProblem): Problem instance
        planner (Planner): Planning algorithm (POUCT or POMCP)
        max_steps (int): Maximum number of steps to run
    """
    total_reward = 0
    
    for step in range(max_steps):
        # Plan
        real_action = planner.plan(maze_problem.agent)

        # Sample next state
        current_state = maze_problem.env.state
        next_state = maze_problem.env.transition_model.sample(current_state, action=real_action)
        
        # Get reward
        reward = maze_problem.env.reward_model.sample(
            current_state, real_action, next_state
        )
        total_reward += reward

        # Get observation from the new state
        real_observation = maze_problem.agent.observation_model.sample(
            next_state, real_action
        )

        # Update history
        maze_problem.agent.update_history(real_action, real_observation)

        # Update belief
        planner.update(maze_problem.agent, real_action, real_observation)

        # Print step information
        print("==== Step %d ====" % (step + 1))
        print("True Position: %s" % str(next_state.position))
        print("True Orientation: %s" % str(next_state.orientation))
        print("Action: %s" % str(real_action))
        print("Observation: %s" % str(real_observation))
        print("Reward: %s" % str(reward))
        print("Cumulative Reward: %s" % str(total_reward))
        
        if isinstance(planner, pomdp_py.POUCT):
            print("Num Sims: %d" % planner.last_num_sims)
            print("Planning Time: %.4fs" % planner.last_planning_time)
        elif isinstance(planner, pomdp_py.POMCP):
            print("Num Sims: %d" % planner.last_num_sims)
            print("Belief Particles: %d" % len(maze_problem.agent.cur_belief))

        # Check if goal reached
        if next_state.position == maze_problem.maze_map.goal:
            print("\n✓ Goal reached!")
            input("Press Enter to continue...")
            break
    
    print(f"\nEpisode complete. Total reward: {total_reward}")


def main():
    """Main solving function"""
    print("Creating Maze Problem...")
    maze = MazeProblem.create()

    print("\n** Testing POUCT **")
    pouct = pomdp_py.POUCT(
        max_depth=10,
        discount_factor=0.95,
        num_sims=1000,
        exploration_const=50,
        rollout_policy=maze.agent.policy_model,
        show_progress=True,
    )
    solve(maze, pouct, max_steps=50)

    # Reset for next test
    # maze = MazeProblem.create()

    # print("\n** Testing POMCP **")
    # maze.agent.set_belief(
    #     pomdp_py.Particles.from_histogram(maze.agent.belief, num_particles=100),
    #     prior=True,
    # )
    # pomcp = pomdp_py.POMCP(
    #     max_depth=10,
    #     discount_factor=0.95,
    #     num_sims=1000,
    #     exploration_const=50,
    #     rollout_policy=maze.agent.policy_model,
    #     show_progress=True,
    #     pbar_update_interval=500,
    # )
    # solve(maze, pomcp, max_steps=50)


if __name__ == "__main__":
    main()
