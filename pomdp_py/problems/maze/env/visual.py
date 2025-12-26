"""
Visualization of the Maze POMDP using pygame.

This module provides interactive visualization of the maze,
showing:
- Maze layout and walls
- Robot's true position (red)
- Goal location (green)
- Start location (yellow)
- Robot's belief distribution (blue particles/histogram)

Run with:
    python -m pomdp_py.problems.maze.env.visual
"""

import pygame
import math
import numpy as np
from pomdp_py.problems.maze.domain.state import State
from pomdp_py.problems.maze.domain.action import get_all_actions
import pomdp_py


class MazeViz:
    """Pygame-based visualization for the Maze POMDP"""

    def __init__(self, maze, cell_size=50, fps=5, controllable=False):
        """
        Initialize the maze visualizer.

        Args:
            maze (MazeProblem): The maze problem instance
            cell_size (int): Size of each grid cell in pixels
            fps (int): Frames per second for the display
            controllable (bool): If True, allow keyboard control
        """
        self.maze = maze
        self.cell_size = cell_size
        self.fps = fps
        self.controllable = controllable
        self.running = False

        # Calculate display size based on maze bounds
        maze_map = maze.maze_map
        # walls format: direction -> list of [(x1,y1), (x2,y2)] segments
        all_points = []
        for segments in maze_map.walls.values():
            for segment in segments:
                for point in segment:
                    all_points.append(point)
        
        self.width = max(p[0] for p in all_points) + 2 if all_points else 12
        self.height = max(p[1] for p in all_points) + 2 if all_points else 8

        self.img_width = self.width * cell_size
        self.img_height = self.height * cell_size

        # Display state
        self._last_action = None
        self._last_observation = None
        self._last_belief = None
        self._step_count = 0

    def on_init(self):
        """Initialize pygame"""
        pygame.init()
        self._display_surf = pygame.display.set_mode(
            (self.img_width + 200, self.img_height), pygame.HWSURFACE
        )
        pygame.display.set_caption("Maze POMDP - Navigation")
        self._background = pygame.Surface(self._display_surf.get_size()).convert()
        self._clock = pygame.time.Clock()
        self._myfont = pygame.font.SysFont("Arial", 20)
        self.running = True

    def on_event(self, event):
        """Handle pygame events"""
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN and self.controllable:
            action = None

            # Keyboard mappings
            if event.key == pygame.K_UP:
                action = "North"
            elif event.key == pygame.K_DOWN:
                action = "South"
            elif event.key == pygame.K_LEFT:
                action = "West"
            elif event.key == pygame.K_RIGHT:
                action = "East"
            elif event.key == pygame.K_SPACE:
                # Random action
                action = None

            if action is not None:
                # Execute action in environment
                from pomdp_py.problems.maze.domain.action import (
                    MoveNorth,
                    MoveSouth,
                    MoveEast,
                    MoveWest,
                )

                action_map = {
                    "North": MoveNorth(),
                    "South": MoveSouth(),
                    "East": MoveEast(),
                    "West": MoveWest(),
                }

                maze_action = action_map[action]
                reward = self.maze.env.state_transition(maze_action, execute=True)
                observation = self.maze.agent.observation_model.sample(
                    self.maze.env.state, maze_action
                )

                self._last_action = maze_action
                self._last_observation = observation
                print(f"Action: {maze_action}, Reward: {reward}, Obs: {observation}")
                self._step_count += 1

    def on_render(self):
        """Render the maze visualization"""
        # Clear background
        self._background.fill((240, 240, 240))

        # Draw maze
        self._draw_maze()

        # Draw goal
        self._draw_point(
            self.maze.maze_map.goal, (0, 200, 0), "G", size=self.cell_size // 2
        )

        # Draw start
        self._draw_point(
            self.maze.maze_map.start, (255, 255, 0), "S", size=self.cell_size // 2
        )

        # Draw belief (if histogram or particles)
        if hasattr(self.maze.agent.cur_belief, "get_histogram"):
            self._draw_belief_histogram()
        elif hasattr(self.maze.agent.cur_belief, "_particles"):
            self._draw_belief_particles()

        # Draw true robot position (red circle)
        true_pos = self.maze.env.state.position
        self._draw_robot(true_pos, self.maze.env.state.orientation)

        # Draw info panel
        self._draw_info_panel()

        # Update display
        pygame.transform.scale(
            self._background,
            (self.img_width + 200, self.img_height),
        )
        self._display_surf.blit(self._background, (0, 0))
        pygame.display.flip()

    def _draw_maze(self):
        """Draw maze walls"""
        maze_map = self.maze.maze_map
        r = self.cell_size

        # Draw grid
        for x in range(self.width):
            for y in range(self.height):
                pygame.draw.rect(
                    self._background,
                    (200, 200, 200),
                    (y * r, x * r, r, r),
                    1,
                )

        # Draw walls
        wall_color = (50, 50, 50)
        wall_thickness = 3

        for direction in ["North", "East", "South", "West"]:
            for segment in maze_map.walls.get(direction, []):
                (x1, y1), (x2, y2) = segment
                start = (y1 * r + r // 2, x1 * r + r // 2)
                end = (y2 * r + r // 2, x2 * r + r // 2)
                pygame.draw.line(self._background, wall_color, start, end, wall_thickness)

    def _draw_point(self, position, color, label="", size=15):
        """Draw a point (goal, start, etc.)"""
        x, y = position
        r = self.cell_size
        center = (y * r + r // 2, x * r + r // 2)
        pygame.draw.circle(self._background, color, center, size)

        # Draw label
        if label:
            text = self._myfont.render(label, True, (0, 0, 0))
            self._background.blit(
                text,
                (center[0] - text.get_width() // 2, center[1] - text.get_height() // 2),
            )

    def _draw_robot(self, position, orientation, size=15):
        """Draw robot with orientation indicator"""
        x, y = position
        r = self.cell_size
        center = (y * r + r // 2, x * r + r // 2)

        # Draw circle
        pygame.draw.circle(self._background, (255, 0, 0), center, size, 2)

        # Draw orientation indicator
        orientation_map = {"North": 0, "East": 90, "South": 180, "West": 270}
        angle = math.radians(orientation_map.get(orientation, 0))
        endpoint = (
            center[0] + int(size * math.cos(angle - math.pi / 2)),
            center[1] + int(size * math.sin(angle - math.pi / 2)),
        )
        pygame.draw.line(self._background, (255, 0, 0), center, endpoint, 2)

    def _draw_belief_histogram(self):
        """Draw belief as histogram (overlapping particles)"""
        histogram = self.maze.agent.cur_belief.get_histogram()

        r = self.cell_size
        colors = [
            (100, 150, 255),  # Light blue
            (0, 100, 255),  # Medium blue
            (0, 0, 200),  # Dark blue
        ]

        # Draw top probability states
        sorted_states = sorted(histogram.items(), key=lambda x: x[1], reverse=True)
        for idx, (state, prob) in enumerate(sorted_states[:10]):
            if prob < 0.01:  # Skip very low probability states
                continue

            x, y = state.position
            center = (y * r + r // 2, x * r + r // 2)
            radius = max(2, int(5 * math.sqrt(prob)))
            color = colors[idx % len(colors)]

            pygame.draw.circle(self._background, color, center, radius)

    def _draw_belief_particles(self):
        """Draw belief as particles"""
        particles = self.maze.agent.cur_belief._particles

        r = self.cell_size

        for particle in particles:
            x, y = particle.position
            center = (y * r + r // 2, x * r + r // 2)
            pygame.draw.circle(self._background, (0, 100, 255), center, 2)

    def _draw_info_panel(self):
        """Draw information panel on the right"""
        panel_x = self.img_width + 10
        panel_y = 10
        line_height = 25

        # Title
        title = self._myfont.render("Maze POMDP", True, (0, 0, 0))
        self._background.blit(title, (panel_x, panel_y))
        panel_y += line_height

        # Current state
        state_text = self._myfont.render(
            f"Position: {self.maze.env.state.position}", True, (0, 0, 0)
        )
        self._background.blit(state_text, (panel_x, panel_y))
        panel_y += line_height

        # Orientation
        orient_text = self._myfont.render(
            f"Orientation: {self.maze.env.state.orientation}", True, (0, 0, 0)
        )
        self._background.blit(orient_text, (panel_x, panel_y))
        panel_y += line_height

        # Last action
        if self._last_action:
            action_text = self._myfont.render(
                f"Last Action: {self._last_action}", True, (0, 0, 0)
            )
            self._background.blit(action_text, (panel_x, panel_y))
        panel_y += line_height

        # Last observation
        if self._last_observation:
            obs_text = self._myfont.render(
                f"Observation: {self._last_observation}", True, (0, 0, 0)
            )
            self._background.blit(obs_text, (panel_x, panel_y))
        panel_y += line_height

        # Belief size
        if hasattr(self.maze.agent.cur_belief, "__len__"):
            belief_text = self._myfont.render(
                f"Belief Size: {len(self.maze.agent.cur_belief)}", True, (0, 0, 0)
            )
            self._background.blit(belief_text, (panel_x, panel_y))
        panel_y += line_height

        # Step count
        step_text = self._myfont.render(f"Steps: {self._step_count}", True, (0, 0, 0))
        self._background.blit(step_text, (panel_x, panel_y))

    def update(self, action, observation):
        """Update visualization with new action and observation"""
        self._last_action = action
        self._last_observation = observation
        self._step_count += 1

    def run_interactive(self):
        """Run interactive visualization (keyboard control)"""
        self.on_init()
        
        while self.running:
            for event in pygame.event.get():
                self.on_event(event)
            
            self.on_render()
            self._clock.tick(self.fps)

        pygame.quit()

    def run_with_planner(self, planner, nsteps=20):
        """Run visualization with a planning algorithm"""
        self.on_init()

        for step in range(nsteps):
            if not self.running:
                break

            # Check for quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break

            if not self.running:
                break

            # Plan
            action = planner.plan(self.maze.agent)

            # Execute - sample next state first
            next_state = self.maze.env.transition_model.sample(
                self.maze.env.state, action
            )
            reward = self.maze.env.reward_model.sample(
                self.maze.env.state, action, next_state
            )
            observation = self.maze.agent.observation_model.sample(
                next_state, action
            )

            # Update history before belief update
            self.maze.agent.update_history(action, observation)

            # Update belief using planner's update method
            planner.update(self.maze.agent, action, observation)

            # Update visualization
            self.update(action, observation)

            # Render
            self.on_render()
            self._clock.tick(self.fps)

            # Check if goal reached
            if next_state.position == self.maze.maze_map.goal:
                print(f"\n🎉 Goal reached in {step + 1} steps!")
                # Display for a few more frames
                for _ in range(10):
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.running = False
                    if not self.running:
                        break
                    self.on_render()
                    self._clock.tick(1)
                break

        pygame.quit()


def main():
    """Main function to demonstrate the visualization"""
    import argparse

    parser = argparse.ArgumentParser(description="Maze POMDP Visualization")
    parser.add_argument(
        "--mode",
        choices=["interactive", "pouct", "pomcp"],
        default="pouct",
        help="Visualization mode",
    )
    parser.add_argument("--fps", type=int, default=2, help="Frames per second")
    parser.add_argument("--nsteps", type=int, default=30, help="Number of steps")

    args = parser.parse_args()

    # Create maze
    maze = MazeProblem.create()

    # Create visualizer
    viz = MazeViz(maze, cell_size=50, fps=args.fps, controllable=(args.mode == "interactive"))

    if args.mode == "interactive":
        print("Interactive Mode - Use arrow keys to control the robot")
        viz.run_interactive()

    elif args.mode == "pouct":
        print("Running with POUCT planner...")
        pouct = pomdp_py.POUCT(
            max_depth=10,
            discount_factor=0.95,
            num_sims=500,
            exploration_const=50,
            rollout_policy=maze.agent.policy_model,
        )
        viz.run_with_planner(pouct, nsteps=args.nsteps)

    elif args.mode == "pomcp":
        print("Running with POMCP planner...")
        maze.agent.set_belief(
            pomdp_py.Particles.from_histogram(maze.agent.belief, num_particles=100),
            prior=True,
        )
        pomcp = pomdp_py.POMCP(
            max_depth=10,
            discount_factor=0.95,
            num_sims=500,
            exploration_const=50,
            rollout_policy=maze.agent.policy_model,
        )
        viz.run_with_planner(pomcp, nsteps=args.nsteps)


if __name__ == "__main__":
    main()
