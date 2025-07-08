# ================================================================================
# This code is adapted from https://github.com/LAVA-LAB/neural_stochastic_control.
# Specifically, models/linearsystem.py.
# ================================================================================

from functools import partial
from itertools import product
from typing import List, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax import jit
from scipy.stats import triang

from core.commons import RectangularSet, MultiRectangularSet
from core.task_decomposition import TaskDecompositionGraph


def create_task_decomposition_nine_rooms():
    nodes = [
        (i, [([0.4 + x, 0.4 + y], [0.6 + x, 0.6 + y])])
        for i, (y, x) in enumerate(product(range(3), repeat=2))
    ]
    edges = [(0, 1), (0, 3), (1, 2), (2, 5), (3, 4), (4, 5), (4, 7), (5, 8), (7, 8)]
    return TaskDecompositionGraph(nodes, edges, 0, 8)


def map_to_subsets(unsafe_map, w, state_space, args):
    if args:
        init = args.target
        target = args.target
        layout = args.layout
    else:
        init = 0
        target = 1
        layout = 1

    # Flip the rows of the map array to match the coordinate system more easily.
    unsafe_map = unsafe_map[::-1]

    unsafe_rectangles = []
    min_x, min_y, max_x, max_y = state_space.low[0], state_space.low[1], state_space.high[0], state_space.high[1]
    for column in range(len(unsafe_map[0])):
        for row in range(len(unsafe_map)):
            x = column // 2
            y = row // 2
            if unsafe_map[row][column] == '+':
                unsafe_rectangles.append(RectangularSet(low=np.array([max(min_x, x - 3 * w), max(min_y, y - w)]),
                                                        high=np.array([min(max_x, x + 3 * w), min(max_y, y + w)]),
                                                        dtype=np.float32))
                unsafe_rectangles.append(RectangularSet(low=np.array([max(min_x, x - w), max(min_y, y - 3 * w)]),
                                                        high=np.array([min(max_x, x + w), min(max_y, y + 3 * w)]),
                                                        dtype=np.float32))
            elif unsafe_map[row][column] == '-':
                unsafe_rectangles.append(RectangularSet(low=np.array([max(min_x, x + 3 * w), max(min_y, y - w)]),
                                                        high=np.array([min(max_x, x + 1 - 3 * w), min(max_y, y + w)]),
                                                        dtype=np.float32))

            elif unsafe_map[row][column] == '|':
                unsafe_rectangles.append(RectangularSet(low=np.array([max(min_x, x - w), max(min_y, y + 3 * w)]),
                                                        high=np.array([min(max_x, x + w), min(max_y, y + 1 - 3 * w)]),
                                                        dtype=np.float32))
            elif unsafe_map[row][column] == 's':
                if layout == 1:
                    init_space = RectangularSet(low=np.array([0.48 + x, 0.48 + y]),
                                                high=np.array([0.52 + x, 0.52 + y]),
                                                dtype=np.float32)
                else:
                    init_space = RectangularSet(low=np.array([0.4 + x, 0.4 + y]), high=np.array([0.6 + x, 0.6 + y]),
                                                dtype=np.float32)
            elif unsafe_map[row][column] == 't':
                target_space = RectangularSet(low=np.array([0.4 + x, 0.4 + y]), high=np.array([0.6 + x, 0.6 + y]),
                                              dtype=np.float32)

    return init_space, target_space, MultiRectangularSet(unsafe_rectangles)


class NineRooms(gym.Env):
    metadata = {
        "render_modes": [],
        "render_fps": 30,
    }

    def __init__(self, args=False):
        self.variable_names = ['x', 'y']
        self.args = args
        self.A = np.diag([1, 1])
        self.B = np.diag([0.1, 0.1])
        if args:
            self.W = np.diag([args.noise, args.noise])
        else:
            self.W = np.diag([0.01, 0.01])

        # Lipschitz coefficient of linear dynamical system is maximum sum of columns in A and B matrix.
        self.lipschitz_f_l1 = float(np.max(np.sum(np.hstack((self.A, self.B)), axis=0)))
        self.lipschitz_f_linfty = float(np.max(np.sum(np.hstack((self.A, self.B)), axis=1)))
        self.lipschitz_f_l1_A = float(np.max(np.sum(self.A, axis=0)))
        self.lipschitz_f_linfty_A = float(np.max(np.sum(self.A, axis=1)))
        self.lipschitz_f_l1_B = float(np.max(np.sum(self.B, axis=0)))
        self.lipschitz_f_linfty_B = float(np.max(np.sum(self.B, axis=1)))

        # Set dimensions
        self.state_dim = 2
        self.noise_dim = 2
        self.plot_dim = self.state_dim
        # Set state space
        low = np.array([0, 0], dtype=np.float32)
        high = np.array([3, 3], dtype=np.float32)
        self.state_space = RectangularSet(low=low, high=high, dtype=np.float32)
        # Set action space
        self.max_torque = np.array([1, 1])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, dtype=np.float32)
        # Set support of noise distribution (which is triangular, zero-centered)
        high = np.array([1, 1], dtype=np.float32)
        self.noise_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        wall_thickness = 0.05

        task_map = (["+-+-+-+",
                     "|    t|",
                     "+-+ + +",
                     "|     |",
                     "+ +-+ +",
                     "|s    |",
                     "+-+-+-+"])

        self.init_space, self.target_space, self.unsafe_space = map_to_subsets(task_map, wall_thickness,
                                                                               self.state_space, args)
        self.init_unsafe_dist = self.init_space.distance_to_sets(self.unsafe_space)

        self.changed_spaces = {}

        self.num_steps_until_reset = 1000

        # Define vectorized functions
        self.vreset = jax.vmap(self.reset_jax, in_axes=0, out_axes=0)
        self.vstep = jax.vmap(self.step_train, in_axes=0, out_axes=0)
        self.vstep_deterministic = jax.vmap(self.step_base, in_axes=0, out_axes=0)

        # Vectorized step, but only with different noise values
        self.vstep_noise_batch = jax.vmap(self.step_noise_key, in_axes=(None, 0, None), out_axes=0)
        self.vstep_noise_set = jax.vmap(self.step_noise_set, in_axes=(None, None, 0, 0), out_axes=(0, 0))

        self.initialize_gym_env()

    # def __copy__(self):
    #     clone = copy.deepcopy(self)
    #     clone._b = some_op(clone._b)
    #     return clone

    def update_task_spaces(self, init=None, target=None, unsafe=None):
        if init is not None:
            self.init_space = init
        if target is not None:
            self.target_space = target
        if unsafe is not None:
            self.unsafe_space = unsafe

        self.init_unsafe_dist = self.init_space.distance_to_sets(self.unsafe_space)

        # Update vectorized functions for new task specification
        self.vreset = jax.vmap(self.reset_jax, in_axes=0, out_axes=0)
        self.vstep = jax.vmap(self.step_train, in_axes=0, out_axes=0)
        self.vstep_deterministic = jax.vmap(self.step_base, in_axes=0, out_axes=0)
        self.vstep_noise_batch = jax.vmap(self.step_noise_key, in_axes=(None, 0, None), out_axes=0)
        self.vstep_noise_set = jax.vmap(self.step_noise_set, in_axes=(None, None, 0, 0), out_axes=(0, 0))

        return

    # def update_changed_spaces(self, changed_space_name: str, changed_space: MultiRectangularSet):
    #     self.changed_spaces[changed_space_name] = changed_space

    def initialize_gym_env(self):
        # Initialize state
        self.state = None
        self.steps_beyond_terminated = None

        # Observation space is only used in the gym version of the environment
        self.observation_space = spaces.Box(low=self.state_space.low, high=self.state_space.high, dtype=np.float32)

    @partial(jit, static_argnums=(0,))
    def sample_noise(self, key, size=None):
        return jax.random.triangular(key, self.noise_space.low * jnp.ones(2), jnp.zeros(self.noise_dim),
                                     self.noise_space.high * jnp.ones(2))

    def sample_noise_numpy(self, size=None):
        return np.random.triangular(self.noise_space.low * np.ones(self.noise_dim), np.zeros(self.noise_dim),
                                    self.noise_space.high * np.ones(self.noise_dim), size)

    @partial(jit, static_argnums=(0,))
    def step_base(self, state, u, w):
        '''
        Make a step in the dynamics. When defining a new environment, this the function that should be modified.
        '''
        u = jnp.clip(u, -self.max_torque, self.max_torque)
        state = jnp.matmul(self.A, state) + jnp.matmul(self.B, u) + jnp.matmul(self.W, w)

        return jnp.clip(state, self.state_space.low, self.state_space.high)

    @partial(jit, static_argnums=(0,))
    def step_noise_set(self, state, u, w_lb, w_ub):
        ''' Make step with dynamics for a set of noise values.
        Propagate state under lower/upper bound of the noise (note: this works because the noise is additive) '''

        # Propagate dynamics for both the lower bound and upper bound of the noise
        # (note: this works because the noise is additive)
        state_lb = self.step_base(state, u, w_lb)
        state_ub = self.step_base(state, u, w_ub)

        # Compute the mean and the epsilon (difference between mean and ub/lb)
        state_mean = (state_ub + state_lb) / 2
        epsilon = (state_ub - state_lb) / 2

        return state_mean, epsilon

    def integrate_noise(self, w_lb, w_ub):
        ''' Integrate noise distribution in the box [w_lb, w_ub]. '''

        # For triangular distribution, integration is simple, because we can integrate each dimension individually and
        # multiply the resulting probabilities
        probs = np.ones(len(w_lb))

        # Triangular cdf increases from loc to (loc + c*scale), and decreases from (loc+c*scale) to (loc + scale)
        # So, 0 <= c <= 1.
        loc = self.noise_space.low
        c = 0.5  # Noise distribution is zero-centered, so c=0.5 by default
        scale = self.noise_space.high - self.noise_space.low

        for d in range(self.noise_space.shape[0]):
            probs *= triang.cdf(w_ub[:, d], c, loc=loc[d], scale=scale[d]) - triang.cdf(w_lb[:, d], c, loc=loc[d],
                                                                                        scale=scale[d])

        # In this case, the noise integration is exact, but we still return an upper and lower bound
        prob_ub = probs
        prob_lb = probs

        return prob_lb, prob_ub

    @partial(jit, static_argnums=(0,))
    def step_noise_key(self, state, key, u):
        # Split RNG key
        key, subkey = jax.random.split(key)

        # Sample noise value
        noise = self.sample_noise(subkey, size=(2,))

        # Propagate dynamics
        state = self.step_base(state, u, noise)

        return state, key

    @partial(jit, static_argnums=(0,))
    def step_train(self, state, key, u, steps_since_reset):
        # Split RNG key
        key, subkey = jax.random.split(key)

        # Sample noise value
        noise = self.sample_noise(subkey, size=(2,))

        goal_reached = self.target_space.jax_contains(jnp.array([state]))[0]
        fail = self.unsafe_space.jax_contains(jnp.array([state]))[0]
        costs = 5 * fail - 5 * goal_reached  # self.target_space.jax_distance_to_center(state) +

        # Propagate dynamics, unless the unsafe region has been entered and the crash argument is true.
        no_crash = 1 - fail * self.args.crash
        state = no_crash * self.step_base(state, u, noise) + (1 - no_crash) * state

        steps_since_reset += 1

        terminated = fail
        truncated = (steps_since_reset >= self.num_steps_until_reset)
        done = terminated | truncated
        state, key, steps_since_reset = self._maybe_reset(state, key, steps_since_reset, done)

        return state, key, steps_since_reset, -costs, terminated, truncated, {}

    def _maybe_reset(self, state, key, steps_since_reset, done):
        return jax.lax.cond(done, self._reset, lambda key: (state, key, steps_since_reset), key)

    def _reset(self, key):
        high = self.state_space.high
        low = self.state_space.low

        key, subkey = jax.random.split(key)
        new_state = jax.random.uniform(subkey, minval=low,
                                       maxval=high, shape=(self.state_dim,))

        steps_since_reset = 0

        return new_state, key, steps_since_reset

    def reset(self, seed=None, options=None):
        ''' Reset function for pytorch / gymnasium environment '''

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Sample state uniformly from observation space
        self.state = np.random.uniform(low=self.observation_space.low, high=self.observation_space.high)
        self.last_u = None

        return self.state, {}

    @partial(jit, static_argnums=(0,))
    def reset_jax(self, key):
        state, key, steps_since_reset = self._reset(key)

        return state, key, steps_since_reset
