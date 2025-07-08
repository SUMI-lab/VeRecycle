# ================================================================================
# This file is adapted from https://github.com/LAVA-LAB/neural_stochastic_control.
# The following methods were added to MultiRectangularSet and RectangularSet:
# - jax_distance_to_border
# - distance_to_border
# - jax_distance_to_center
# - distance_to_center
# - distance_to_sets
# ================================================================================

import time
from functools import partial
from typing import Self

import control as ct
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces


class MultiRectangularSet:
    '''
    Class to create a set of rectangular sets.
    '''

    def __init__(self, sets):
        self.sets = sets
        self.dimension = sets[0].dimension
        self.change_dimensions = sets[0].change_dimensions

    def get_volume(self):
        return np.sum([Set.volume for Set in self.sets])

    def contains(self, xvector, dim=-1, delta=0, return_indices=False):

        # Remove the extra columns from the data (storing additional data beyond the grid points)
        if dim != -1:
            xvector_trim = xvector[:, :dim]
        else:
            xvector_trim = xvector

        # bools[x] = 1 if x is contained in set
        bools = np.array([set.contains(xvector_trim, delta=delta, return_indices=True) for set in self.sets])

        # Point is contained if it is contained in any of the sets
        bools = np.any(bools, axis=0)

        if return_indices:
            return bools
        else:
            return xvector[bools]

    @partial(jax.jit, static_argnums=(0))
    def jax_contains(self, xvector, delta=0):

        # bools[x] = 1 if x is contained in set
        bools = jnp.array([set.jax_contains(xvector, delta) for set in self.sets])

        # Point is contained if it is contained in any of the sets
        bools = jnp.any(bools, axis=0)

        return bools

    def not_contains(self, xvector, dim=-1, delta=0, return_indices=False):

        # Remove the extra columns from the data (storing additional data beyond the grid points)
        if dim != -1:
            xvector_trim = xvector[:, :dim]
        else:
            xvector_trim = xvector

        # bools[x] = 1 if x is *not* contained in set
        bools = np.array([set.not_contains(xvector_trim, delta=delta, return_indices=True) for set in self.sets])

        # Point is not contained if it is contained in none of the sets
        bools = np.all(bools, axis=0)

        if return_indices:
            return bools
        else:
            return xvector[bools]

    @partial(jax.jit, static_argnums=(0))
    def jax_not_contains(self, xvector, delta=0):

        # bools[x] = 1 if x is *not* contained in set
        bools = jnp.array([set.jax_not_contains(xvector, delta) for set in self.sets])

        # Point is not contained if it is contained in none of the sets
        bools = jnp.all(bools, axis=0)

        return bools

    @partial(jax.jit, static_argnums=(0, 2))
    def sample(self, rng, N, delta=0):
        # Sample n values for each of the state sets and return the stacked vector
        rngs = jax.random.split(rng, len(N))
        samples = [Set.sample(rng, n, delta) for (Set, rng, n) in zip(self.sets, rngs, N)]
        samples = jnp.vstack(samples)

        return samples

    @partial(jax.jit, static_argnums=(0))
    def sample_single(self, rng):
        # First determine from which initial state set to take a sample
        rng, subkey = jax.random.split(rng)

        # First sample one state from each set
        samples = jnp.vstack([Set.sample_single(rng) for Set in self.sets])

        # Then randomly return one of them
        sample = jax.random.choice(subkey, samples)

        return sample

    @partial(jax.jit, static_argnums=(0,))
    def jax_distance_to_border(self, point):
        return jnp.min(jnp.array([rect.distance_to_border(point) for rect in self.sets]))

    def distance_to_border(self, point):
        return np.min([rect.distance_to_border(point) for rect in self.sets])

    @partial(jax.jit, static_argnums=(0,))
    def jax_distance_to_center(self, point):
        return jnp.min(jnp.array([rect.distance_to_center(point) for rect in self.sets]))

    def distance_to_center(self, point):
        return np.min([rect.distance_to_center(point) for rect in self.sets])

    def distance_to_sets(self, rectangles):
        return np.min([rect.distance_to_sets(rectangles) for rect in self.sets])


class RectangularSet:
    '''
    Class to create a rectangular set with cheap containment checks (faster than gymnasium Box.contains).
    '''

    def __init__(self, low, high, change_dimensions=False, dtype=np.float32):

        self.low = np.array(low, dtype=dtype)
        self.high = np.array(high, dtype=dtype)
        self.gymspace = spaces.Box(low=low, high=high, dtype=dtype)
        self.dimension = len(self.low)
        self.volume = np.prod(self.high - self.low)
        if not change_dimensions:
            self.change_dimensions = np.ones_like(self.low)
        else:
            self.change_dimensions = np.ones_like(self.low)
            self.change_dimensions[np.array(change_dimensions)] = 0

    def distance_to_border(self, point):
        """
        Calculate the Euclidean distance between a point and a Rectangular set.

        Args:
            point (tuple): a point.

        Returns:
            distance (float): the Euclidean distance between the point and the closest point on the boundary of
            the Rectangular set.
        """

        assert len(point) == self.dimension, (f"Cannot calculate the distance between a point with dim {len(point)} "
                                              f"and a box with {self.dimension}.")

        s = 0
        for i in range(self.dimension):
            s += np.max([self.low[i] - point[i], 0, point[i] - self.high[i]]) ** 2

        return np.sqrt(s)

    def distance_to_sets(self, rectangles: Self | MultiRectangularSet):
        """
        Calculate the Euclidean distance between a MultiRectangularSet and a Rectangular set.

        Args:
            rectangles (MultiRectangularSet): a set of RectangularSet objects.

        Returns:
            distance (float): the Euclidean distance between the border of this RectangularSet and the border of the
            closest RectangularSet from MultiRectangularSet rectangles.
        """
        if isinstance(rectangles, RectangularSet):
            rectangles = MultiRectangularSet([rectangles])

        min_distance = np.inf
        for rectangle in rectangles.sets:
            d1 = self.low - rectangle.high
            d2 = rectangle.low - self.high

            u = np.max(np.array([np.zeros(len(d1)), d1]), axis=0)
            v = np.max(np.array([np.zeros(len(d2)), d2]), axis=0)

            dist = np.linalg.norm(np.concatenate([u, v]))
            min_distance = min(min_distance, dist)

        return min_distance

    def distance_to_center(self, point):
        """
        Calculate the Euclidean distance between a point and the center of a Rectangular set.

        Args:
            point (tuple): a point.

        Returns:
            distance (float): the Euclidean distance between the point and the center of the Rectangular set.
        """

        assert len(point) == self.dimension, (f"Cannot calculate the distance between a point with dim {len(point)} "
                                              f"and a box with {self.dimension}.")

        s = 0
        for i in range(self.dimension):
            center = (self.low[i] + self.high[i]) / 2
            s += (point[i] - center) ** 2

        return np.sqrt(s)

    @partial(jax.jit, static_argnums=(0,))
    def jax_distance_to_border(self, point):
        """
        Calculate the Euclidean distance between a point and a Rectangular set.

        Args:
            point (tuple): a point.

        Returns:
            distance (float): the Euclidean distance between the point and the closest point on the boundary of the
            Rectangular set.
        """

        s = 0
        for i in range(self.dimension):
            s += jnp.max(jnp.array([self.low[i] - point[i], 0, point[i] - self.high[i]])) ** 2

        return jnp.sqrt(s)

    @partial(jax.jit, static_argnums=(0,))
    def jax_distance_to_center(self, point):
        """
        Calculate the Euclidean distance between a point and the center of a Rectangular set.

        Args:
            point (tuple): a point.

        Returns:
            distance (float): the Euclidean distance between the point and the center the Rectangular set.
        """
        s = 0
        for i in range(self.dimension):
            center = (self.low[i] + self.high[i]) / 2
            s += (point[i] - center) ** 2

        return jnp.sqrt(s)

    def get_volume(self):
        return self.volume

    def contains(self, xvector, dim=-1, delta=0, return_indices=False):
        '''
        Check if a vector of points is contained in the rectangular set, expanded by a value of delta.
        :param xvector: vector of points
        :param delta: expand by
        :return: list of booleans
        '''

        # Remove the extra columns from the data (storing additional data beyond the grid points)
        if dim != -1:
            xvector_trim = xvector[:, :dim]
        else:
            xvector_trim = xvector

        delta_dims = np.kron(delta, self.change_dimensions.reshape(-1, 1)).T

        # Note: we actually want to check that x >= low - delta, but we rewrite this to avoid issues with dimensions
        # caused by numpy (same for the other expression).
        bools = np.all((xvector_trim + delta_dims) >= self.low, axis=1) * \
                np.all((xvector_trim - delta_dims) <= self.high, axis=1)

        if return_indices:
            return bools
        else:
            return xvector[bools]

    @partial(jax.jit, static_argnums=(0,))
    def jax_contains(self, xvector, delta=0):
        '''
        Check if a vector of points is contained in the rectangular set.
        :param xvector: vector of points
        :param delta: expand by
        '''

        delta_dims = jnp.kron(delta, self.change_dimensions.reshape(-1, 1)).T

        bools = jnp.all(xvector >= self.low - delta_dims, axis=1) * \
                jnp.all(xvector <= self.high + delta_dims, axis=1)
        return bools

    def not_contains(self, xvector, dim=-1, delta=0, return_indices=False):
        '''
        Check if a vector of points is *not* contained in the rectangular set, expanded by a value of delta.
        :param xvector: vector of points
        :param delta: expand by
        :return: list of booleans
        '''

        # Remove the extra columns from the data (storing additional data beyond the grid points)
        if dim != -1:
            xvector_trim = xvector[:, :dim]
        else:
            xvector_trim = xvector

        delta_dims = np.kron(delta, self.change_dimensions.reshape(-1, 1)).T

        # Note: we actually want to check that x < low - delta, but we rewrite this to avoid issues with dimensions
        # caused by numpy (same for the other expression).
        bools = np.any((xvector_trim + delta_dims) < self.low, axis=1) + \
                np.any((xvector_trim - delta_dims) > self.high, axis=1)

        if return_indices:
            return bools
        else:
            return xvector[bools]

    @partial(jax.jit, static_argnums=(0,))
    def jax_not_contains(self, xvector, delta=0):
        '''
        Check if a vector of points is *not* contained in the rectangular set.
        :param xvector: vector of points
        :param delta: expand by
        '''

        delta_dims = jnp.kron(delta, self.change_dimensions.reshape(-1, 1)).T

        # Note: we actually want to check that x < low - delta, but we rewrite this to avoid issues with dimensions
        # caused by numpy (same for the other expression).
        bools = jnp.any(xvector < self.low - delta_dims, axis=1) + \
                jnp.any(xvector > self.high + delta_dims, axis=1)
        return bools

    @partial(jax.jit, static_argnums=(0, 2))
    def sample(self, rng, N, delta=0):
        # Uniformly sample n values from this state set
        samples = jax.random.uniform(rng, (N, self.dimension), minval=self.low - delta, maxval=self.high + delta)

        return samples

    @partial(jax.jit, static_argnums=(0))
    def sample_single(self, rng):

        # Uniformly sample n values from this state set
        sample = jax.random.uniform(rng, (self.dimension,), minval=self.low, maxval=self.high)

        return sample


def lqr(A, B, Q, R, verbose=False):
    K, S, E = ct.dlqr(A, B, Q, R)

    if verbose:
        print('Eigenvalues of closed-loop system:', E)
        print('Control gain matrix:', K)

    return K


def TicTocGenerator():
    ''' Generator that returns the elapsed run time '''
    ti = time.time()  # initial time
    tf = time.time()  # final time
    while True:
        tf = time.time()
        yield tf - ti  # returns the time difference


def TicTocDifference():
    ''' Generator that returns time differences '''
    tf0 = time.time()  # initial time
    tf = time.time()  # final time
    while True:
        tf0 = tf
        tf = time.time()
        yield tf - tf0  # returns the time difference


TicToc = TicTocGenerator()  # create an instance of the TicTocGen generator
TicTocDiff = TicTocDifference()  # create an instance of the TicTocGen generator


def toc(tempBool=True):
    ''' Print current time difference '''
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print("Elapsed time: %f seconds." % tempTimeInterval)


def tic():
    ''' Start time recorder '''
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


def tocDiff(tempBool=True):
    ''' Print current time difference '''
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicTocDiff)
    if tempBool:
        print("Elapsed time: %f seconds.\n" % np.round(tempTimeInterval, 5))
    else:
        return np.round(tempTimeInterval, 12)

    return tempTimeInterval


def ticDiff():
    ''' Start time recorder '''
    # Records a time in TicToc, marks the beginning of a time interval
    tocDiff(False)


def args2dict(**kwargs):
    ''' Return all arguments passed to the function as a dictionary. '''
    return locals()['kwargs']


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
