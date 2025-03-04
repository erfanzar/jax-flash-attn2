import logging

from jax import random as jrandom
from jax.interpreters import pxla
from jax.sharding import PartitionSpec


class GenerateRNG:
	"""An infinite generator of JAX PRNGKeys, useful for iterating over seeds."""

	def __init__(self, seed: int = 0):
		"""Initializes the generator with a starting seed.

		Args:
		    seed: The seed to use for the initial PRNGKey.
		"""
		self.seed = seed
		self._rng = jrandom.PRNGKey(seed)

	def __next__(self) -> jrandom.PRNGKey:
		"""Generates and returns the next PRNGKey in the sequence.

		Returns:
		    The next PRNGKey derived from the internal state.
		"""
		self._rng, key = jrandom.split(self._rng)
		return key

	@property
	def rng(self) -> jrandom.PRNGKey:
		"""Provides access to the next PRNGKey without advancing the generator.

		Returns:
		    The next PRNGKey in the sequence.
		"""
		return next(self)


def get_logger(name, level: int = logging.INFO) -> logging.Logger:
	"""
	Function to create and configure a logger.
	Args:
	    name: str: The name of the logger.
	    level: int: The logging level. Defaults to logging.INFO.
	Returns:
	    logging.Logger: The configured logger instance.
	"""
	logger = logging.getLogger(name)
	logger.propagate = False
	logger.setLevel(level)
	console_handler = logging.StreamHandler()
	console_handler.setLevel(level)
	formatter = logging.Formatter("%(asctime)s %(levelname)-8s [%(name)s] %(message)s")
	console_handler.setFormatter(formatter)
	logger.addHandler(console_handler)
	return logger


def names_in_current_mesh(*names: str) -> bool:
	"""
	Check if the given names are present in the current JAX mesh.

	Args:
	    *names: Variable number of axis names to check.

	Returns:
	    True if all given names are present in the current mesh, False otherwise.
	"""
	mesh_axis_names = pxla.thread_resources.env.physical_mesh.axis_names
	return set(names) <= set(mesh_axis_names)


def get_names_from_partition_spec(
	partition_specs: dict[str, PartitionSpec],
) -> list[str]:
	"""
	Extract axis names from a partition specification.

	This function recursively iterates through the provided `partition_specs`
	dictionary and extracts all unique axis names used in the sharding specifications.

	Args:
	    partition_specs: A dictionary mapping parameter names to their respective `PartitionSpec`.

	Returns:
	    A list of unique axis names used in the partition specs.
	"""
	names = set()
	if isinstance(partition_specs, dict):
		partition_specs = partition_specs.values()
	for item in partition_specs:
		if item is None:
			continue
		elif isinstance(item, str):
			names.add(item)
		else:
			names.update(get_names_from_partition_spec(item))
	return list(names)
