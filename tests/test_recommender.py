from numpy import matrix
from recommender import __version__

from recommender.models.matrix_factorization import MatrixFactorization
import jax


def test_version():
    assert __version__ == "0.1.0"


def test_mf():

    model = MatrixFactorization(num_users=5, num_items=6, features=4)

    inputs = {
        "user_ids": jax.numpy.array([1, 2, 3]),
        "item_ids": jax.numpy.array([1, 2, 3]),
    }

    out, params = model.init_with_output(jax.random.PRNGKey(0), inputs)

    assert out.shape == (3,)
