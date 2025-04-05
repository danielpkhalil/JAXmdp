# gymnax_env_framestack.py

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, Tuple, Optional, Union
from flax import struct
import chex
from gymnax.environments import environment, spaces

@struct.dataclass
class TabularState(environment.EnvState):
    """
    We add a frame_buffer field to store stacked frames if num_frames > 1.
    If num_frames = 1 (the default), this buffer can remain empty (size=0).
    """
    state_idx: jnp.int32
    steps: jnp.int32
    done: bool
    time: int
    # Holds the stacked frames with shape [H, W, 3 * num_frames] if screens exist
    frame_buffer: chex.Array = jnp.array([])  # empty by default

@struct.dataclass
class TabularEnvParams(environment.EnvParams):
    done_on_reward: bool = struct.field(default=False, pytree_node=False)
    no_done_reward: float = struct.field(default=0.0, pytree_node=False)
    use_screen_observations: bool = struct.field(default=True, pytree_node=False)
    horizon: int = struct.field(default=40, pytree_node=False)
    max_steps_in_episode: int = struct.field(default=40, pytree_node=False)
    reward_scale: float = struct.field(default=1.0, pytree_node=False)
    # --- NEW PARAMETER: number of frames to stack ---
    num_frames: int = struct.field(default=1, pytree_node=False)

class TabularEnv(environment.Environment):
    def __init__(self, problem_file: str):
        super().__init__()
        mdp = np.load(problem_file, allow_pickle=True, mmap_mode="r")
        self.num_states, self._num_actions = mdp["transitions"].shape
        self.transitions = jnp.array(mdp["transitions"])
        self.rewards = jnp.array(mdp["rewards"])

        self.screens = None
        self.screen_mapping = None
        if "screens" in mdp:
            self.screens = jnp.array(mdp["screens"])
        if "screen_mapping" in mdp:
            self.screen_mapping = jnp.array(mdp["screen_mapping"])

        self.TERMINAL_STATE = -1

    @property
    def name(self) -> str:
        return "TabularMDP"

    @property
    def num_actions(self) -> int:
        return self._num_actions

    def default_params(self) -> TabularEnvParams:
        return TabularEnvParams()

    def action_space(self, params: Optional[TabularEnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: Optional[TabularEnvParams] = None) -> spaces.Space:
        """
        If num_frames > 1 and we have screen observations, the channel dimension is multiplied
        by num_frames. Otherwise, the space is unchanged.
        """
        if params is None:
            params = self.default_params()

        if params.use_screen_observations and (self.screens is not None):
            base_shape = self.screens.shape[1:]  # e.g. (H, W, 3)
            if params.num_frames > 1:
                # Multiply the last (channel) dimension by num_frames
                stacked_channels = base_shape[-1] * params.num_frames
                stacked_shape = base_shape[:-1] + (stacked_channels,)
                return spaces.Box(
                    low=0,
                    high=255,
                    shape=stacked_shape,
                    dtype=jnp.uint8,
                )
            else:
                # Original single-frame shape
                return spaces.Box(
                    low=0,
                    high=255,
                    shape=base_shape,
                    dtype=jnp.uint8,
                )
        else:
            # Tabular observation is just the state_idx in a 1D array
            return spaces.Box(
                low=0,
                high=self.num_states - 1,
                shape=(1,),
                dtype=jnp.float32,
            )

    def state_space(self, params: TabularEnvParams) -> spaces.Dict:
        return spaces.Dict(
            {
                "state_idx": spaces.Discrete(self.num_states),
                "steps": spaces.Discrete(params.horizon),
                "done": spaces.Discrete(2),
                "time": spaces.Discrete(params.max_steps_in_episode),
                # frame_buffer is not explicitly described in spaces
                # since it's an internal detail of the environment state.
            }
        )

    def reset_env(
        self,
        key: chex.PRNGKey,
        params: Optional[TabularEnvParams] = None
    ) -> Tuple[chex.Array, TabularState]:
        """
        Initialize the environment state and frame buffer if num_frames > 1.
        """
        if params is None:
            params = self.default_params()

        init_state = TabularState(
            state_idx=jnp.int32(0),
            steps=jnp.int32(0),
            done=False,
            time=0,
            frame_buffer=jnp.array([])  # empty by default
        )

        # Get the single-frame observation
        init_obs = self._get_single_frame_obs(init_state, params)

        # If we’re stacking frames, initialize frame_buffer by repeating init_obs
        if (
            params.use_screen_observations
            and self.screens is not None
            and params.num_frames > 1
        ):
            # init_obs shape is (H, W, 3)
            repeated = jnp.concatenate([init_obs] * params.num_frames, axis=-1)
            init_state = init_state.replace(frame_buffer=repeated)

            # The returned observation is the stacked buffer
            return init_state.frame_buffer, init_state
        else:
            # num_frames=1 or not using screens => just return the single observation
            return init_obs, init_state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: TabularState,
        action: Union[int, float, chex.Array],
        params: Optional[TabularEnvParams] = None
    ) -> Tuple[chex.Array, TabularState, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """
        Advance the environment by one step, then build the stacked observation
        if num_frames>1.
        """
        if params is None:
            params = self.default_params()

        def if_done_fn(_):
            reward = jnp.float32(0.0)
            # Even if done, we typically return the last known observation
            obs_if_done = self.get_obs(state, params)
            info = {
                "steps": state.steps,
                "reward": reward,
                "done_by_terminal": jnp.array(False),
                "done_by_horizon": jnp.array(False),
                "done_by_reward": jnp.array(False),
                "discount": self.discount(state, params),
            }
            return obs_if_done, state, reward, state.done, info

        def if_not_done_fn(_):
            next_state_idx = self.transitions[state.state_idx, action]
            reward = self.rewards[state.state_idx, action]

            new_steps = state.steps + 1
            done_by_terminal = (next_state_idx == self.TERMINAL_STATE)
            done_by_horizon = (new_steps >= params.horizon)
            done_by_reward = (reward != 0) & (params.done_on_reward)
            done_new = done_by_terminal | done_by_horizon | done_by_reward

            # Possibly add no_done_reward if we hit horizon without terminal
            reward += jnp.where(
                done_by_horizon & (~done_by_terminal),
                jnp.float32(params.no_done_reward),
                jnp.float32(0),
            )
            # Scale the reward
            reward *= jnp.float32(params.reward_scale)

            next_state = TabularState(
                state_idx=jnp.where(done_new, state.state_idx, next_state_idx),
                steps=new_steps,
                done=done_new,
                time=state.time + 1,
                # We'll handle the frame_buffer below
                frame_buffer=state.frame_buffer
            )

            # Build single-frame observation
            next_obs_single = self._get_single_frame_obs(next_state, params)

            # Update the frame buffer if needed
            if (
                params.use_screen_observations
                and self.screens is not None
                and params.num_frames > 1
            ):
                # Shift old buffer, drop oldest frame chunk, and append new
                num_channels = self.screens.shape[-1]  # typically 3
                old_buffer = next_state.frame_buffer

                def init_buffer_fn(_):
                    # If for some reason it's empty (like first step), replicate next_obs
                    return jnp.concatenate([next_obs_single] * params.num_frames, axis=-1)

                def update_buffer_fn(buf):
                    # Remove the oldest frame from the left, append new frame on the right
                    # old_buffer shape: (H, W, 3 * num_frames)
                    # next_obs_single shape: (H, W, 3)
                    return jnp.concatenate(
                        [buf[..., num_channels:], next_obs_single],
                        axis=-1
                    )

                new_buffer = jax.lax.cond(
                    old_buffer.size == 0,  # i.e., is it empty?
                    init_buffer_fn,
                    update_buffer_fn,
                    operand=old_buffer
                )

                next_state = next_state.replace(frame_buffer=new_buffer)
                stacked_obs = new_buffer
            else:
                # num_frames=1 or no screens => single-frame observation
                stacked_obs = next_obs_single

            info = {
                "steps": new_steps,
                "reward": reward,
                "done_by_terminal": jnp.array(done_by_terminal),
                "done_by_horizon": jnp.array(done_by_horizon),
                "done_by_reward": jnp.array(done_by_reward),
                "discount": self.discount(next_state, params),
            }
            return stacked_obs, next_state, reward, jnp.array(done_new), info

        return jax.lax.cond(state.done, if_done_fn, if_not_done_fn, operand=None)

    def get_obs(
        self,
        state: TabularState,
        params: Optional[TabularEnvParams] = None,
        key: Optional[chex.PRNGKey] = None
    ) -> chex.Array:
        """
        Returns the current observation, which may be single-frame or stacked
        if num_frames>1. For convenience, we rely on the state's frame_buffer
        when frames are being stacked.
        """
        if params is None:
            params = self.default_params()

        if (
            params.use_screen_observations
            and self.screens is not None
            and params.num_frames > 1
        ):
            # If we're framestacking, the state.frame_buffer is always the
            # final stacked observation
            return state.frame_buffer
        else:
            # Otherwise, just return the single-frame observation
            return self._get_single_frame_obs(state, params)

    def _get_single_frame_obs(
        self,
        state: TabularState,
        params: TabularEnvParams
    ) -> chex.Array:
        """
        Internal helper that returns only the single current frame or tabular state index.
        This is used both in reset_env and step_env to build the stacked frames.
        """
        if params.use_screen_observations and self.screens is not None:
            def valid_screen_fn(idx):
                return self.screens[self.screen_mapping[idx]]

            def invalid_screen_fn(_):
                return jnp.zeros(self.screens.shape[1:], dtype=jnp.uint8)

            return jax.lax.cond(
                (state.state_idx >= 0) & (state.state_idx < self.num_states),
                valid_screen_fn,
                invalid_screen_fn,
                state.state_idx
            )
        else:
            # Tabular observation: single integer in a 1D array
            return jnp.array([state.state_idx], dtype=jnp.float32)

    def discount(self, state: TabularState, params: Optional[TabularEnvParams] = None) -> jnp.ndarray:
        return jnp.array(1.0 - state.done, dtype=jnp.float32)

    def is_terminal(self, state: TabularState, params: TabularEnvParams) -> jnp.ndarray:
        return state.done
