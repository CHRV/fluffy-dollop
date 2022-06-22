import random

import jax
import optax
from flax.core import freeze
from flax.core.frozen_dict import FrozenDict


def zero_grads():
    # from https://github.com/deepmind/optax/issues/159#issuecomment-896459491
    def init_fn(_):
        return ()

    def update_fn(updates, state, params=None):
        return jax.tree_map(jax.numpy.zeros_like, updates), ()

    return optax.GradientTransformation(init_fn, update_fn)


def create_mask(params):
    use_user = random.choice([True, False])

    def label_fn(k):
        if use_user:
            print("user")
            return k.startswith("user")

        else:
            print("item")
            return k.startswith("item")

    def _map(params, mask):
        for k in params:
            if label_fn(k):
                mask[k] = "zero"
            else:
                if isinstance(params[k], FrozenDict):
                    mask[k] = {}
                    _map(params[k], mask[k])
                else:
                    mask[k] = "adam"

    mask = {}
    _map(params, mask)
    return freeze(mask)


tx = optax.multi_transform(
    {"adam": optax.adam(1e-2), "zero": optax.set_to_zero()}, create_mask
)
