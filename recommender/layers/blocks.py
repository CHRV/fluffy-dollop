from typing import List, Optional
import flax.linen as nn
import jax


class MLP(nn.Module):
    units: List[int]
    activation: Optional[str] = "relu"

    @nn.compact
    def __call__(self, x):
        for unit in self.units[:-1]:
            x = nn.linear.Dense(unit)(x)
            x = jax.nn.relu(x)

        x = nn.linear.Dense(self.units[-1])(x)

        return x
