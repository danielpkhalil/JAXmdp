import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, Tuple, Optional, Union
from flax import struct
import chex
from gymnax.environments import environment, spaces

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
    framestack: int = struct.field(default=1, pytree_node=False)

class TabularEnv(environment.Environment):
    """
    This environment:
      - If framestack=1, returns (H, W, 3) frames as usual.
      - If framestack>1 and 'prev_state_mapping' is present,
        it returns (H, W, 3 * framestack).
      - We do NOT store frames in TabularState, so there's
        no shape mismatch in 'env_state' across steps.
      - Instead, we reconstruct stacked frames by looking up
        'prev_state_mapping' from the current state_idx backwards.
    """
    def __init__(self, problem_file: str):
        super().__init__()
        mdp = np.load(problem_file, allow_pickle=True, mmap_mode="r")
        self.num_states, self._num_actions = mdp["transitions"].shape

        self.transitions = jnp.array(mdp["transitions"])
        self.rewards = jnp.array(mdp["rewards"])
        self.screens = None
        self.screen_mapping = None
        self.prev_state_mapping = None

        if "screens" in mdp:
            self.screens = jnp.array(mdp["screens"])  # shape: (num_distinct_screens, H, W, 3)
        if "screen_mapping" in mdp:
            self.screen_mapping = jnp.array(mdp["screen_mapping"])  # shape: (num_states,)
        if "prev_state_mapping" in mdp:
            self.prev_state_mapping = jnp.array(mdp["prev_state_mapping"])  # shape: (num_states,)

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
        Returns either:
          - (H, W, 3) if framestack=1
          - (H, W, 3 * framestack) if framestack>1
          - or shape (1,) if no screens are present.
        """
        if params is None:
            params = self.default_params()

        # If using screen observations and we have screens
        if params.use_screen_observations and (self.screens is not None):
            h, w, c = self.screens.shape[1:]  # e.g. (H, W, 3)
            if params.framestack > 1 and (self.prev_state_mapping is not None):
                shape = (h, w, c * params.framestack)  # e.g. (H, W, 3 * 4)
            else:
                shape = (h, w, c)  # single frame
            return spaces.Box(low=0, high=255, shape=shape, dtype=jnp.uint8)
        else:
            # Fallback: a single scalar state index
            return spaces.Box(low=0, high=self.num_states - 1, shape=(1,), dtype=jnp.float32)

    def state_space(self, params: TabularEnvParams) -> spaces.Dict:
        return spaces.Dict({
            "state_idx": spaces.Discrete(self.num_states),
            "steps": spaces.Discrete(params.horizon),
            "done": spaces.Discrete(2),
            "time": spaces.Discrete(params.max_steps_in_episode),
        })

    def reset_env(
        self,
        key: chex.PRNGKey,
        params: Optional[TabularEnvParams] = None
    ) -> Tuple[chex.Array, TabularState]:
        if params is None:
            params = self.default_params()
        init_state = TabularState(
            state_idx=jnp.array(0, dtype=jnp.int32),
            steps=jnp.array(0, dtype=jnp.int32),
            done=jnp.array(False),
            time=0
        )
        init_obs = self.get_obs(init_state, params)
        return init_obs, init_state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: TabularState,
        action: Union[int, float, chex.Array],
        params: Optional[TabularEnvParams] = None
    ) -> Tuple[chex.Array, TabularState, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        if params is None:
            params = self.default_params()

        def if_done_fn(_):
            reward = jnp.array(0.0, dtype=jnp.float32)
            next_obs = self.get_obs(state, params)
            info = {
                "steps": state.steps,
                "reward": reward,
                "done_by_terminal": jnp.array(False),
                "done_by_horizon": jnp.array(False),
                "done_by_reward": jnp.array(False),
                "discount": self.discount(state, params),
            }
            return next_obs, state, reward, state.done, info

        def if_not_done_fn(_):
            next_state_idx = self.transitions[state.state_idx, action]
            reward = self.rewards[state.state_idx, action]

            new_steps = state.steps + 1
            done_by_terminal = (next_state_idx == self.TERMINAL_STATE)
            done_by_horizon = (new_steps >= params.horizon)
            done_by_reward = (reward != 0) & (params.done_on_reward)
            done_new = done_by_terminal | done_by_horizon | done_by_reward

            # possible 'no_done_reward'
            reward += jnp.where(
                done_by_horizon & ~done_by_terminal,
                jnp.float32(params.no_done_reward),
                jnp.float32(0),
            )

            # Scale reward
            reward = reward * jnp.float32(params.reward_scale)

            # If done, stay in the same state_idx
            next_state = TabularState(
                state_idx=jnp.where(done_new, state.state_idx, next_state_idx),
                steps=new_steps,
                done=done_new,
                time=state.time + 1
            )
            next_obs = self.get_obs(next_state, params)

            info = {
                "steps": new_steps,
                "reward": reward,
                "done_by_terminal": jnp.array(done_by_terminal),
                "done_by_horizon": jnp.array(done_by_horizon),
                "done_by_reward": jnp.array(done_by_reward),
                "discount": self.discount(next_state, params),
            }
            return next_obs, next_state, reward, jnp.array(done_new), info

        return jax.lax.cond(state.done, if_done_fn, if_not_done_fn, operand=None)

    def get_obs(
        self,
        state: TabularState,
        params: Optional[TabularEnvParams] = None,
        key: Optional[chex.PRNGKey] = None
    ) -> chex.Array:
        """
        Return a single screen of shape (H, W, 3) if framestack=1,
        or a stacked set of screens (H, W, 3 * framestack) by traversing
        prev_state_mapping up to 'framestack' times.
        If no screens exist, we fallback to the integer state index.
        """
        if params is None:
            params = self.default_params()

        # If screens are present and user wants them
        if params.use_screen_observations and self.screens is not None:
            # If we have framestack>1 and a prev_state_mapping
            if params.framestack > 1 and self.prev_state_mapping is not None:
                # Gather up to 'framestack' states: current + previous
                def gather_screen(idx):
                    # If idx is valid, return that screen
                    valid_idx = (idx >= 0) & (idx < self.num_states)
                    return jnp.where(
                        valid_idx,
                        self.screens[self.screen_mapping[idx]],  # shape (H,W,3)
                        jnp.zeros(self.screens.shape[1:], dtype=jnp.uint8),
                    )

                idx_list = [state.state_idx]
                cur_idx = state.state_idx
                # walk backward
                for _ in range(params.framestack - 1):
                    cur_idx = jnp.where(
                        (cur_idx >= 0) & (cur_idx < self.num_states),
                        self.prev_state_mapping[cur_idx],
                        -1
                    )
                    idx_list.append(cur_idx)

                # Reverse so oldest is first
                idx_list = idx_list[::-1]

                # For each idx in idx_list, gather (H,W,3), then concatenate along channels
                stacked_frames = jnp.concatenate(
                    [gather_screen(i) for i in idx_list], axis=-1
                )
                # shape => (H, W, 3 * framestack)
                return stacked_frames

            else:
                # Single-frame
                def valid_screen_fn(idx):
                    return self.screens[self.screen_mapping[idx]]

                def invalid_screen_fn(_):
                    return jnp.zeros(self.screens.shape[1:], dtype=jnp.uint8)

                return jax.lax.cond(
                    (state.state_idx >= 0) & (state.state_idx < self.num_states),
                    valid_screen_fn,
                    invalid_screen_fn,
                    state.state_idx,
                )

        # else fallback to integer index
        return jnp.array([state.state_idx], dtype=jnp.float32)

    def discount(self, state: TabularState, params: Optional[TabularEnvParams] = None) -> jnp.ndarray:
        return jnp.array(1.0 - state.done, dtype=jnp.float32)

    def is_terminal(self, state: TabularState, params: TabularEnvParams) -> jnp.ndarray:
        return state.done
