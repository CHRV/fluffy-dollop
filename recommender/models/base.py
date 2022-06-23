from abc import ABC, abstractmethod
from functools import partial

import jax


class Model(ABC):
    @abstractmethod
    def compute_loss(self, inputs, targets, training: bool = False):
        """Defines the loss function.
        Args:
          inputs: A data structure of tensors: raw inputs to the model. These will
            usually contain labels and weights as well as features.
          training: Whether the model is in training mode.
        Returns:
          Loss tensor.
        """
        pass

    @partial(jax.jit, static_argnames=["self"])
    def train_step(self, inputs, targets):
        """Custom train step using the `compute_loss` method."""

        loss, grads = self.compute_loss(inputs, targets, training=True)

        self.state = self.state.apply_gradients(grads=grads)
        metrics = {}
        # metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss

        return metrics

    def test_step(self, inputs, targets):
        """Custom train step using the `compute_loss` method."""
        loss = self.compute_loss(inputs, targets, training=False)

        metrics = {}
        metrics["loss"] = loss

        return metrics
