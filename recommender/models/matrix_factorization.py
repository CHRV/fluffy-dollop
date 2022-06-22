from typing import Any, Callable

import einops
import flax.linen as nn
from matplotlib.pyplot import axis
import optax
from flax.core.scope import VariableDict
from flax.training.train_state import TrainState
import jax
from jax import numpy as jnp
from recommender.models import base


class MatrixFactorization(nn.Module):
    num_users: int
    num_items: int
    features: int

    def setup(self):
        self.user_embed = nn.linear.Embed(self.num_users, self.features)
        self.user_bias_embed = nn.linear.Embed(self.num_users, 1)

        self.item_embed = nn.linear.Embed(self.num_items, self.features)
        self.item_bias_embed = nn.linear.Embed(self.num_items, 1)

    def __call__(self, inputs):
        user_batch = inputs["user_ids"]
        item_batch = inputs["item_ids"]

        assert user_batch.shape == item_batch.shape

        user_embedding = self.user_embed(user_batch)
        user_bias = self.user_bias_embed(user_batch)

        item_embedding = self.item_embed(item_batch)
        item_bias = self.item_bias_embed(item_batch)
        ratings_logits = jnp.sum(user_embedding * item_embedding, axis=1)
        ratings_logits = ratings_logits + jnp.squeeze(user_bias + item_bias)

        return jax.nn.sigmoid(ratings_logits)


class MatrixFactorizationModel(base.Model):
    model: MatrixFactorization
    state: TrainState
    loss_fn: Callable[[Any, Any], float]

    def __init__(
        self,
        model: MatrixFactorization,
        params: VariableDict,
        loss_fn: Callable,
        optim=optax.adam(1e-2),
    ) -> None:
        self._model = model
        self.state = TrainState.create(apply_fn=model.apply, params=params, tx=optim)
        self.loss_fn = loss_fn

    def compute_loss(self, inputs, targets, training: bool = False):
        def loss(params):
            out = self.state.apply_fn(params, inputs)
            return self.loss_fn(targets, out)

        if training:
            return jax.value_and_grad(loss)(self.state.params)
        return loss(self.state.params)
