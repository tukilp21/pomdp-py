import pomdp_py


class MazeEnvironment(pomdp_py.Environment):
    """
    The environment for the maze domain.
    It maintains the true state of the robot and manages state transitions.
    """

    def __init__(self, init_state, transition_model, reward_model, maze_map):
        """
        Args:
            init_state: Initial state of the robot
            transition_model: Transition model for state transitions
            reward_model: Reward model
            maze_map: MazeMap object containing the maze layout
        """
        super().__init__(init_state, transition_model, reward_model)
        self.maze_map = maze_map

    def state_transition(self, action, execute=True):
        """
        Execute an action in the environment and return the reward.
        
        Args:
            action: Action to execute
            execute: If True, updates the environment state
            
        Returns:
            reward: Reward from the transition
        """
        if execute:
            # Sample next state from transition model
            next_state = self.transition_model.sample(self.state, action)
            # Get reward
            reward = self.reward_model.sample(self.state, action, next_state)
            # Update state
            self.state = next_state
            return reward
        else:
            # Just compute reward without updating
            next_state = self.transition_model.sample(self.state, action)
            reward = self.reward_model.sample(self.state, action, next_state)
            return reward
