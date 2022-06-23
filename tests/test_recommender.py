import jax
import optax
from recommender import __version__
from recommender.models.matrix_factorization import (
    MatrixFactorization,
    MatrixFactorizationModel,
)

from .helpers import tx


def test_version():
    assert __version__ == "0.1.0"


def test_mf():

    model = MatrixFactorization(num_users=5, num_items=6, features=4)

    inputs = {
        "user_ids": jax.numpy.array([1, 2, 3]),
        "item_ids": jax.numpy.array([1, 2, 3]),
    }

    out, _ = model.init_with_output(jax.random.PRNGKey(0), inputs)

    assert out.shape == (3,)


def test_mf_model():

    mf = MatrixFactorization(num_users=5, num_items=6, features=4)
    inputs = {
        "user_ids": jax.numpy.array([1, 2, 3]),
        "item_ids": jax.numpy.array([1, 2, 3]),
    }

    params = mf.init(jax.random.PRNGKey(0), inputs)

    model = MatrixFactorizationModel(
        mf, params, lambda x, y: jax.numpy.sum((x - y) ** 2), optax.adam(1e-3)
    )

    for _ in range(5):
        model.train_step(inputs, jax.numpy.array([0] * 3))
