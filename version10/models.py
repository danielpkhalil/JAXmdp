import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import orthogonal
import distrax

# ------------------------------
# IMPALA CNN Building Blocks
# ------------------------------
class ResidualBlock(nn.Module):
    """A simple 2-conv residual block (no batch norm), each conv 3x3 stride=1."""
    out_channels: int

    @nn.compact
    def __call__(self, x):
        # Residual branch
        residual = x

        # First conv
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_init=orthogonal(np.sqrt(2))
        )(x)
        x = nn.relu(x)

        # Second conv
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_init=orthogonal(np.sqrt(2))
        )(x)

        # Residual add
        x = x + residual
        x = nn.relu(x)
        return x


class ImpalaBlock(nn.Module):
    """One 'IMPALA block': residual block, then downsampling by 2."""
    out_channels: int

    @nn.compact
    def __call__(self, x):
        # If the input channel doesn't match out_channels, we project it
        in_channels = x.shape[-1]
        if in_channels != self.out_channels:
            x = nn.Conv(
                features=self.out_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                kernel_init=orthogonal(1.0),
                padding="VALID",
            )(x)

        # One or more residual blocks
        x = ResidualBlock(self.out_channels)(x)
        x = ResidualBlock(self.out_channels)(x)

        # Downsample by 2
        # Typical IMPALA uses AvgPool(3x3, stride=2, same padding)
        x = nn.avg_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        return x


# ------------------------------
# IMPALA CNN Actor-Critic
# ------------------------------
class ImpalaCNNActorCritic(nn.Module):
    """
    Three IMPALA blocks, each doubling channel depth per the typical usage:
      out_channels = [16, 32, 32] (adjust as you wish)
    Then flatten, feed into FC of size 256 (or your choice),
    produce separate policy (Categorical logits) and value.
    """
    action_dim: int
    conv_channels: tuple = (16, 32, 32)
    fc_output_dim: int = 256

    @nn.compact
    def __call__(self, obs):
        """
        obs shape: (batch_size, H, W, C). Typically, for Atari: (batch, 84, 84, 4),
        scaled 0..255. We'll normalize below if desired.
        """
        # Convert to float32
        x = obs.astype(jnp.float32)
        # Optional: scale inputs to [0,1], depends on your pipeline
        # x = x / 255.0

        # Run blocks
        for out_c in self.conv_channels:
            x = ImpalaBlock(out_c)(x)

        # Flatten
        x = x.reshape((x.shape[0], -1))
        # FC
        x = nn.Dense(
            features=self.fc_output_dim,
            kernel_init=orthogonal(np.sqrt(2))
        )(x)
        x = nn.relu(x)

        # Policy head
        logits = nn.Dense(
            features=self.action_dim,
            kernel_init=orthogonal(0.01)
        )(x)
        pi = distrax.Categorical(logits=logits)

        # Value head
        value = nn.Dense(
            features=1,
            kernel_init=orthogonal(1.0)
        )(x)
        return pi, jnp.squeeze(value, axis=-1)
