"""
Tabular MDP → Gymnax wrapper
============================
Loads an NPZ bundle with
    • transitions   (S, A)
    • rewards       (S, A)
    • screens, screen_mapping  (optional)
Adds **optional action masking**:
    mask[i] = 1  ⇔  action i changes state
                    (self-loops are the ONLY invalid actions)
Return format
-------------
use_action_mask = False  →  obs  = uint8[H,W,3] or float32[1]
use_action_mask = True   →  obs  = {
                                "obs"        : <image | vector>,
                                "action_mask": float32[A]  (0/1)
                             }
The tiny `step` override is there because the generic Gymnax helper
expects array observations only.
"""

from typing import Any, Dict, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import environment, spaces


# ------------------------------------------------------------------ #
#  Dataclasses                                                       #
# ------------------------------------------------------------------ #
@struct.dataclass
class TabularState(environment.EnvState):
    state_idx: jnp.int32
    steps: jnp.int32
    done: bool
    time: int


@struct.dataclass
class TabularEnvParams(environment.EnvParams):
    done_on_reward: bool = struct.field(default=False, pytree_node=False)
    no_done_reward: float = struct.field(default=0.0, pytree_node=False)
    use_screen_observations: bool = struct.field(default=True, pytree_node=False)
    horizon: int = struct.field(default=40, pytree_node=False)
    max_steps_in_episode: int = struct.field(default=40, pytree_node=False)
    reward_scale: float = struct.field(default=1.0, pytree_node=False)

    # flag to switch masking on/off
    use_action_mask: bool = struct.field(default=False, pytree_node=False)


# ------------------------------------------------------------------ #
#  Environment                                                       #
# ------------------------------------------------------------------ #
class TabularEnv(environment.Environment):
    TERMINAL_STATE = -1

    def __init__(self, problem_file: str):
        super().__init__()
        mdp = np.load(problem_file, allow_pickle=True, mmap_mode="r")
        self.num_states, self._num_actions = mdp["transitions"].shape

        self.transitions = jnp.array(mdp["transitions"])
        self.rewards = jnp.array(mdp["rewards"])

        self.screens = jnp.array(mdp["screens"])
        self.screen_mapping = jnp.array(mdp["screen_mapping"])

    # ------------------------------------------------------------------ #
    #  Helper: action mask                                               #
    # ------------------------------------------------------------------ #
    def get_action_mask(
        self,
        state: TabularState,
        params: Optional[TabularEnvParams] = None,
    ) -> jnp.ndarray:
        """Valid = action that leads to a NEW state (terminal moves allowed)."""
        next_vec = self.transitions[state.state_idx]          # (A,)
        valid = next_vec != state.state_idx                   # ban self-loops
        return valid.astype(jnp.float32)

    # ------------------------------------------------------------------ #
    #  Spaces                                                             #
    # ------------------------------------------------------------------ #
    def default_params(self) -> TabularEnvParams:
        return TabularEnvParams()

    def action_space(self, params=None) -> spaces.Discrete:
        return spaces.Discrete(self._num_actions)

    def observation_space(self, params=None) -> spaces.Space:
        if params is None:
            params = self.default_params()

        # base observation space
        if params.use_screen_observations:
            base = spaces.Box(
                low=0, high=255, shape=self.screens.shape[1:], dtype=jnp.uint8
            )
        else:
            base = spaces.Box(
                low=0, high=self.num_states - 1, shape=(1,), dtype=jnp.float32
            )

        if params.use_action_mask:
            return spaces.Dict(
                {
                    "obs": base,
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(self._num_actions,), dtype=jnp.float32
                    ),
                }
            )
        else:
            return base

    def state_space(self, params) -> spaces.Dict:
        return spaces.Dict(
            dict(
                state_idx=spaces.Discrete(self.num_states),
                steps=spaces.Discrete(params.horizon),
                done=spaces.Discrete(2),
                time=spaces.Discrete(params.max_steps_in_episode),
            )
        )

    # ------------------------------------------------------------------ #
    #  Reset                                                              #
    # ------------------------------------------------------------------ #
    def reset_env(self, key, params=None):
        if params is None:
            params = self.default_params()

        init_state = TabularState(
            state_idx=jnp.array(0, dtype=jnp.int32),
            steps=jnp.array(0, dtype=jnp.int32),
            done=jnp.array(False),
            time=0,
        )

        base_obs = self.get_obs(init_state, params)
        if params.use_action_mask:
            obs = {"obs": base_obs, "action_mask": self.get_action_mask(init_state, params)}
        else:
            obs = base_obs
        return obs, init_state

    # ------------------------------------------------------------------ #
    #  Step (core logic)                                                  #
    # ------------------------------------------------------------------ #
    def step_env(self, key, state, action, params=None):
        if params is None:
            params = self.default_params()

        # --------- done branch ---------- #
        def if_done(_):
            base_obs = self.get_obs(state, params)
            obs = (
                {"obs": base_obs, "action_mask": self.get_action_mask(state, params)}
                if params.use_action_mask
                else base_obs
            )
            reward = jnp.float32(0.0)
            info = dict(
                steps=state.steps,
                reward=reward,
                done_by_terminal=False,
                done_by_horizon=False,
                done_by_reward=False,
                discount=self.discount(state, params),
            )
            return obs, state, reward, state.done, info

        # --------- normal transition ---- #
        def if_not_done(_):
            next_idx = self.transitions[state.state_idx, action]
            reward = self.rewards[state.state_idx, action]

            new_steps = state.steps + 1
            done_by_terminal = next_idx == self.TERMINAL_STATE
            done_by_horizon = new_steps >= params.horizon
            done_by_reward = (reward != 0) & params.done_on_reward
            done_new = done_by_terminal | done_by_horizon | done_by_reward

            reward += jnp.where(
                done_by_horizon & ~done_by_terminal,
                jnp.float32(params.no_done_reward),
                0.0,
            )
            reward *= jnp.float32(params.reward_scale)

            next_state = TabularState(
                state_idx=jnp.where(done_new, state.state_idx, next_idx),
                steps=new_steps,
                done=done_new,
                time=state.time + 1,
            )

            base_obs = self.get_obs(next_state, params)
            obs = (
                {"obs": base_obs, "action_mask": self.get_action_mask(next_state, params)}
                if params.use_action_mask
                else base_obs
            )

            info = dict(
                steps=new_steps,
                reward=reward,
                done_by_terminal=done_by_terminal,
                done_by_horizon=done_by_horizon,
                done_by_reward=done_by_reward,
                discount=self.discount(next_state, params),
            )
            return obs, next_state, reward, done_new, info

        return jax.lax.cond(state.done, if_done, if_not_done, operand=None)

    # override – avoids arrays-only helper
    def step(self, key, state, action, params=None):
        return self.step_env(key, state, action, params)

    # ------------------------------------------------------------------ #
    #  Observation helper                                                 #
    # ------------------------------------------------------------------ #
    def get_obs(self, state, params=None, key=None):
        if params is None:
            params = self.default_params()

        if params.use_screen_observations:
            def valid(idx):
                return self.screens[self.screen_mapping[idx]]

            def invalid(_):
                return jnp.zeros(self.screens.shape[1:], dtype=jnp.uint8)

            return jax.lax.cond(
                (state.state_idx >= 0) & (state.state_idx < self.num_states),
                valid,
                invalid,
                state.state_idx,
            )
        else:
            return jnp.array([state.state_idx], dtype=jnp.float32)

    def discount(self, state, params=None):
        return jnp.array(1.0 - state.done, dtype=jnp.float32)

    def is_terminal(self, state, params):
        return state.done
