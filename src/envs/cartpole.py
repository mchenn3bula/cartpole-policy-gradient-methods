"""Cartpole environment."""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding


class CartpoleEnv(gym.Env):
    """Cartpole environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, env_context=None, render_mode=None):
        """Initialize environment.

        Args:
            env_context (dict): environment configuration.
            render_mode (str): render mode.
        """
        # Variables
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # Actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # Seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max,
        ], dtype=np.float32)

        # Action and observation (state) spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Render mode
        self.render_mode = render_mode

        # Others
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

    def _process_state(self):
        """Process state before returning it.

        Returns:
            state_processed (numpy.array): processed state.
        """
        # No modifications
        processed_state = self.state

        # ---> TODO: remove accelerations from state

        # ---> TODO: if no accelerations, determine a new working state

        return processed_state

    def reset(self, seed=None, options=None):
        """Reset the environment.

        Args:
            seed (int): seed for reproducibility.
            options (dict): additional information.

        Returns:
            state (numpy.array): the processed state.

            info (dict): auxiliary diagnostic information.
        """
        # Reset seed
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        # Current time step
        self._time_step = 0

        # Reset state
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state = self.state.astype(np.float32)

        # ---> TODO: if no accelerations, determine a new working state

        # Eventually render
        if self.render_mode == "human":
            self.render()

        return self._process_state(), {}

    def step(self, action):
        """Go from current step to next one.

        Args:
            action (int): action of the agent.

        Returns:
            state (numpy.array): state.

            reward (float): reward.

            terminated (bool): whether a terminal state is reached.

            truncated (bool): whether a truncation condition is reached.

            info (dict): auxiliary diagnostic information.
        """
        # Check if action is valid
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        # Compute variables
        x_tmp = self.state
        x, x_dot, theta, theta_dot = x_tmp[0], x_tmp[1], x_tmp[2], x_tmp[3]
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # https://coneural.org/florian/papers/05_cart_pole.pdf
        m = self.polemass_length
        temp = force + m * theta_dot**2 * sintheta / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp)
        thetaacc /= 4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass
        thetaacc /= self.length
        xacc = temp - m * thetaacc * costheta / self.total_mass

        # Update system
        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # Semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        # ---> TODO: if no accelerations, determine a new working state

        # Full system state
        self.state = np.array([
            x,
            x_dot,
            theta,
            theta_dot,
        ], dtype=np.float32)

        # Reward is 1
        reward = 1.0

        # Increase time step
        self._time_step += 1

        # Check if episode if finished
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self._time_step >= 500
        )

        # Eventually render
        if self.render_mode == "human":
            self.render()

        return self._process_state(), reward, terminated, False, {}

    def render(self):
        """Render environment.

        Note:
            Do not pay too much attention to this function. It is just to
            display a nice animation of the environment.
        """
        import pygame
        from pygame import gfxdraw

        # Initialize render mode if needed
        if self.render_mode is None:
            self.render_mode = "human"

        # Initialize objects
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height))
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface(
                    (self.screen_width, self.screen_height))

        # Initialize clock
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Objects
        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        # Get state
        if self.state is None:
            return None
        x = self.state

        # Get surface
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        # Computations
        l = -cartwidth / 2
        r = cartwidth / 2
        t = cartheight / 2
        b = -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(self.surf,
                         int(cartx),
                         int(carty + axleoffset),
                         int(polewidth / 2),
                         (129, 132, 203))
        gfxdraw.filled_circle(self.surf,
                              int(cartx),
                              int(carty + axleoffset),
                              int(polewidth / 2),
                              (129, 132, 203))

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        # Display
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        # Human mode
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        # RGB array mode
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)),
                axes=(1, 0, 2)
            )

    def close(self):
        """Close the environment.

        Note:
            Do not pay too much attention to this function. It is just to close
            the environment.
        """
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
