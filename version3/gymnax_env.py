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
    time: int  # Required by base class


@struct.dataclass
class TabularEnvParams(environment.EnvParams):
    done_on_reward: bool = False
    no_done_reward: float = 0.0
    use_screen_observations: bool = True
    horizon: int = 10000
    max_steps_in_episode: int = 10000  # Required by base class


class TabularEnv(environment.Environment):
    """
    A gymnax-compatible environment for a tabular MDP stored in an .npz file.
    """

    def __init__(self, problem_file: str):
        super().__init__()
        # Load the .npz file
        mdp = np.load(problem_file, allow_pickle=True, mmap_mode="r")

        self.num_states, self._num_actions = mdp["transitions"].shape

        # Convert to jax.numpy for functional usage
        self.transitions = jnp.array(mdp["transitions"])  # shape: (S, A)
        self.rewards = jnp.array(mdp["rewards"])  # shape: (S, A)

        # Check for optional keys
        self.screens = None
        self.screen_mapping = None
        if "screens" in mdp and "screen_mapping" in mdp:
            self.screens = jnp.array(mdp["screens"])  # shape: (num_screens, H, W, 3)
            self.screen_mapping = jnp.array(mdp["screen_mapping"])  # shape: (num_states,)

        self.TERMINAL_STATE = -1

    @property
    def default_params(self) -> TabularEnvParams:
        """Default environment parameters"""
        return TabularEnvParams()

    @property
    def name(self) -> str:
        """Environment name."""
        return "TabularMDP"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment"""
        return self._num_actions

    def action_space(self, params: Optional[TabularEnvParams] = None) -> spaces.Discrete:
        """Action space of the environment"""
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: Optional[TabularEnvParams] = None) -> spaces.Space:
        """Observation space of the environment"""
        if params is None:
            params = self.default_params

        if params.use_screen_observations and (self.screens is not None):
            return spaces.Box(
                low=0,
                high=255,
                shape=self.screens.shape[1:],  # (H, W, 3)
                dtype=jnp.uint8
            )
        else:
            return spaces.Discrete(self.num_states)

    def state_space(self, params: TabularEnvParams) -> spaces.Dict:
        """State space of the environment"""
        if params is None:
            params = self.default_params

        return spaces.Dict({
            "state_idx": spaces.Discrete(self.num_states),
            "steps": spaces.Discrete(params.horizon),
            "done": spaces.Discrete(2),
            "time": spaces.Discrete(params.max_steps_in_episode)
        })

    def reset_env(
            self,
            key: chex.PRNGKey,
            params: Optional[TabularEnvParams] = None
    ) -> Tuple[chex.Array, TabularState]:
        """Reset environment to initial state"""
        del key  # Not used in deterministic MDP
        if params is None:
            params = self.default_params

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
        """Execute step in environment"""
        del key  # Not used unless you have random transitions
        if params is None:
            params = self.default_params

        # If we're already done, remain in the same state with zero reward
        def if_done_fn(_):
            next_state = TabularState(
                state_idx=state.state_idx,
                steps=state.steps,
                done=state.done,
                time=state.time
            )
            reward = jnp.array(0.0, dtype=jnp.float32)
            done = jnp.array(state.done)
            next_obs = self.get_obs(state, params)
            info = {
                "steps": state.steps,
                "reward": reward,
                "done_by_terminal": jnp.array(False),
                "done_by_horizon": jnp.array(False),
                "done_by_reward": jnp.array(False),
                "discount": self.discount(state, params)
            }
            return next_obs, next_state, reward, done, info

        # If not done, do the usual MDP transition
        def if_not_done_fn(_):
            next_state_idx = self.transitions[state.state_idx, action]
            reward = self.rewards[state.state_idx, action]

            new_steps = state.steps + 1
            done_by_terminal = (next_state_idx == self.TERMINAL_STATE)
            done_by_horizon = (new_steps >= params.horizon)
            done_by_reward = (reward != 0) & params.done_on_reward

            done_new = done_by_terminal | done_by_horizon | done_by_reward

            reward += jnp.where(
                done_by_horizon & ~done_by_terminal,
                jnp.float32(params.no_done_reward),
                jnp.float32(0)
            )

            next_state = TabularState(
                state_idx=jnp.where(done_new, state.state_idx, next_state_idx),
                steps=new_steps,
                done=done_new,
                time=state.time + 1
            )
            next_obs = self.get_obs(next_state, params)

            # Convert all boolean flags to jnp.arrays
            info = {
                "steps": new_steps,
                "reward": jnp.array(reward, dtype=jnp.float32),
                "done_by_terminal": jnp.array(done_by_terminal),
                "done_by_horizon": jnp.array(done_by_horizon),
                "done_by_reward": jnp.array(done_by_reward),
                "discount": self.discount(next_state, params)
            }
            return next_obs, next_state, jnp.array(reward, dtype=jnp.float32), jnp.array(done_new), info

        return jax.lax.cond(
            state.done,
            if_done_fn,
            if_not_done_fn,
            operand=None
        )

    def get_obs(
            self,
            state: TabularState,
            params: Optional[TabularEnvParams] = None,
            key: Optional[chex.PRNGKey] = None
    ) -> chex.Array:
        """Get observation from state"""
        if params is None:
            params = self.default_params

        if params.use_screen_observations and (self.screens is not None):
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
            return jnp.array(state.state_idx, dtype=jnp.int32)

    def discount(self, state: TabularState, params: Optional[TabularEnvParams] = None) -> jnp.ndarray:
        """Return discount factor for current state."""
        return jnp.array(1.0 - state.done, dtype=jnp.float32)


def create_tabular_env(problem_file: str) -> TabularEnv:
    """Create a TabularEnv from a given .npz file path."""
    return TabularEnv(problem_file)







# import jax
# import jax.numpy as jnp
# import numpy as np
# from typing import Any, Dict, Tuple, Optional, Union
# from flax import struct
# import chex
# from gymnax.environments import environment, spaces
#
#
# @struct.dataclass
# class TabularState(environment.EnvState):
#     state_idx: jnp.int32
#     steps: jnp.int32
#     done: bool
#
#
# @struct.dataclass
# class TabularEnvParams(environment.EnvParams):
#     done_on_reward: bool = False
#     no_done_reward: float = 0.0
#     use_screen_observations: bool = False
#     horizon: int = 10000
#     max_steps_in_episode: int = 10000
#
#
# class TabularEnv(environment.Environment):
#     """
#     A gymnax-compatible environment for a tabular MDP stored in an .npz file.
#     """
#
#     def __init__(self, problem_file: str):
#         super().__init__()
#         # Load the .npz file
#         mdp = np.load(problem_file, allow_pickle=True)
#
#         self.num_states, self._num_actions = mdp["transitions"].shape
#
#         # Convert to jax.numpy for functional usage
#         self.transitions = jnp.array(mdp["transitions"])  # shape: (S, A)
#         self.rewards = jnp.array(mdp["rewards"])  # shape: (S, A)
#
#         # Check for optional keys
#         self.screens = None
#         self.screen_mapping = None
#         if "screens" in mdp and "screen_mapping" in mdp:
#             self.screens = jnp.array(mdp["screens"])  # shape: (num_screens, H, W, 3)
#             self.screen_mapping = jnp.array(mdp["screen_mapping"])  # shape: (num_states,)
#
#         self.TERMINAL_STATE = -1
#
#     @property
#     def default_params(self) -> TabularEnvParams:
#         """Default environment parameters"""
#         return TabularEnvParams()
#
#     @property
#     def name(self) -> str:
#         """Environment name."""
#         return "TabularMDP"
#
#     @property
#     def num_actions(self) -> int:
#         """Number of actions possible in environment"""
#         return self._num_actions
#
#     def action_space(self, params: Optional[TabularEnvParams] = None) -> spaces.Discrete:
#         """Action space of the environment"""
#         return spaces.Discrete(self.num_actions)
#
#     def observation_space(self, params: Optional[TabularEnvParams] = None) -> spaces.Space:
#         """Observation space of the environment"""
#         if params is None:
#             params = self.default_params
#
#         if params.use_screen_observations and (self.screens is not None):
#             return spaces.Box(
#                 low=0,
#                 high=255,
#                 shape=self.screens.shape[1:],  # (H, W, 3)
#                 dtype=jnp.uint8
#             )
#         else:
#             return spaces.Discrete(self.num_states)
#
#     def state_space(self, params: TabularEnvParams) -> spaces.Dict:
#         """State space of the environment"""
#         if params is None:
#             params = self.default_params
#
#         return spaces.Dict({
#             "state_idx": spaces.Discrete(self.num_states),
#             "steps": spaces.Discrete(params.horizon),
#             "done": spaces.Discrete(2)
#         })
#
#     def reset_env(
#             self,
#             key: chex.PRNGKey,
#             params: Optional[TabularEnvParams] = None
#     ) -> Tuple[chex.Array, TabularState]:
#         """Reset environment to initial state"""
#         del key  # Not used in deterministic MDP
#         if params is None:
#             params = self.default_params
#
#         init_state = TabularState(
#             state_idx=jnp.array(0, dtype=jnp.int32),
#             steps=jnp.array(0, dtype=jnp.int32),
#             done=jnp.array(False)
#         )
#         init_obs = self.get_obs(init_state, params)
#         return init_obs, init_state
#
#     def step_env(
#             self,
#             key: chex.PRNGKey,
#             state: TabularState,
#             action: Union[int, float, chex.Array],
#             params: Optional[TabularEnvParams] = None
#     ) -> Tuple[chex.Array, TabularState, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
#         """Execute step in environment"""
#         del key  # Not used unless you have random transitions
#         if params is None:
#             params = self.default_params
#
#         # If we're already done, remain in the same state with zero reward
#         def if_done_fn(_):
#             next_state = state
#             reward = jnp.array(0.0, dtype=jnp.float32)
#             done = jnp.array(state.done)
#             next_obs = self.get_obs(state, params)
#             info = {
#                 "steps": state.steps,
#                 "reward": reward,
#                 "done_by_terminal": jnp.array(False),
#                 "done_by_horizon": jnp.array(False),
#                 "done_by_reward": jnp.array(False),
#                 "discount": self.discount(state, params)
#             }
#             return next_obs, next_state, reward, done, info
#
#         # If not done, do the usual MDP transition
#         def if_not_done_fn(_):
#             next_state_idx = self.transitions[state.state_idx, action]
#             reward = self.rewards[state.state_idx, action]
#
#             new_steps = state.steps + 1
#             done_by_terminal = (next_state_idx == self.TERMINAL_STATE)
#             done_by_horizon = (new_steps >= params.horizon)
#             done_by_reward = (reward != 0) & params.done_on_reward
#
#             done_new = done_by_terminal | done_by_horizon | done_by_reward
#
#             reward += jnp.where(
#                 done_by_horizon & ~done_by_terminal,
#                 jnp.float32(params.no_done_reward),
#                 jnp.float32(0)
#             )
#
#             next_state = TabularState(
#                 state_idx=jnp.where(done_new, state.state_idx, next_state_idx),
#                 steps=new_steps,
#                 done=done_new
#             )
#             next_obs = self.get_obs(next_state, params)
#
#             # Convert all boolean flags to jnp.arrays
#             info = {
#                 "steps": new_steps,
#                 "reward": jnp.array(reward, dtype=jnp.float32),
#                 "done_by_terminal": jnp.array(done_by_terminal),
#                 "done_by_horizon": jnp.array(done_by_horizon),
#                 "done_by_reward": jnp.array(done_by_reward),
#                 "discount": self.discount(next_state, params)
#             }
#             return next_obs, next_state, jnp.array(reward, dtype=jnp.float32), jnp.array(done_new), info
#
#         return jax.lax.cond(
#             state.done,
#             if_done_fn,
#             if_not_done_fn,
#             operand=None
#         )
#
#     def get_obs(
#             self,
#             state: TabularState,
#             params: Optional[TabularEnvParams] = None,
#             key: Optional[chex.PRNGKey] = None
#     ) -> chex.Array:
#         """Get observation from state"""
#         if params is None:
#             params = self.default_params
#
#         if params.use_screen_observations and (self.screens is not None):
#             def valid_screen_fn(idx):
#                 return self.screens[self.screen_mapping[idx]]
#
#             def invalid_screen_fn(_):
#                 return jnp.zeros(self.screens.shape[1:], dtype=jnp.uint8)
#
#             return jax.lax.cond(
#                 (state.state_idx >= 0) & (state.state_idx < self.num_states),
#                 valid_screen_fn,
#                 invalid_screen_fn,
#                 state.state_idx
#             )
#         else:
#             return jnp.array(state.state_idx, dtype=jnp.int32)
#
#     def discount(self, state: TabularState, params: Optional[TabularEnvParams] = None) -> jnp.ndarray:
#         """Return discount factor for current state."""
#         return jnp.array(1.0 - state.done, dtype=jnp.float32)
#
#
# def create_tabular_env(problem_file: str) -> TabularEnv:
#     """Create a TabularEnv from a given .npz file path."""
#     return TabularEnv(problem_file)










# import jax
# import jax.numpy as jnp
# import numpy as np
# from typing import NamedTuple, Any, Dict, Tuple, Optional
# from gymnax.environments import environment, spaces
#
#
# class TabularState(NamedTuple):
#     """Holds the JAX-compatible state for our tabular environment."""
#     state_idx: jnp.int32
#     steps: jnp.int32
#     done: bool
#
#
# class TabularEnv(environment.Environment):
#     """
#     A gymnax-compatible environment for a tabular MDP stored in an .npz file.
#     """
#
#     def __init__(self, problem_file: str):
#         super().__init__()
#         # Load the .npz file
#         mdp = np.load(problem_file, allow_pickle=True)
#
#         self.num_states, self._num_actions = mdp["transitions"].shape
#
#         # Convert to jax.numpy for functional usage
#         self.transitions = jnp.array(mdp["transitions"])  # shape: (S, A)
#         self.rewards = jnp.array(mdp["rewards"])         # shape: (S, A)
#
#         # Check for optional keys
#         self.screens = None
#         self.screen_mapping = None
#         if "screens" in mdp and "screen_mapping" in mdp:
#             self.screens = jnp.array(mdp["screens"])                # shape: (num_screens, H, W, 3)
#             self.screen_mapping = jnp.array(mdp["screen_mapping"])  # shape: (num_states,)
#
#         self.TERMINAL_STATE = -1
#
#     @property
#     def default_params(self) -> Dict[str, Any]:
#         """Default environment parameters"""
#         return {
#             "done_on_reward": False,
#             "no_done_reward": 0.0,
#             "use_screen_observations": False,
#             "horizon": 10000
#         }
#
#     @property
#     def num_actions(self) -> int:
#         """Number of actions possible in environment"""
#         return self._num_actions
#
#     @property
#     def action_space(self) -> spaces.Discrete:
#         """Action space of the environment"""
#         return spaces.Discrete(self.num_actions)
#
#     @property
#     def observation_space(self) -> spaces.Space:
#         """Observation space of the environment"""
#         if self.default_params["use_screen_observations"] and (self.screens is not None):
#             return spaces.Box(
#                 low=0,
#                 high=255,
#                 shape=self.screens.shape[1:],  # (H, W, 3)
#                 dtype=jnp.uint8
#             )
#         else:
#             return spaces.Discrete(self.num_states)
#
#     def reset_env(
#         self,
#         key: jax.random.PRNGKey,
#         params: Optional[Dict[str, Any]] = None
#     ) -> Tuple[TabularState, jnp.ndarray]:
#         """Reset environment to initial state"""
#         del key  # Not used in deterministic MDP
#         if params is None:
#             params = self.default_params
#
#         init_state = TabularState(
#             state_idx=jnp.int32(0),
#             steps=jnp.int32(0),
#             done=False
#         )
#         init_obs = self.get_obs(init_state, params)
#         return init_state, init_obs
#
#     def step_env(
#             self,
#             key: jax.random.PRNGKey,
#             state: TabularState,
#             action: jnp.int32,
#             params: Optional[Dict[str, Any]] = None
#     ) -> Tuple[TabularState, jnp.float32, bool, jnp.ndarray, Dict[str, Any]]:
#         """Execute step in environment"""
#         del key  # Not used unless you have random transitions
#         if params is None:
#             params = self.default_params
#
#         # If we're already done, remain in the same state with zero reward
#         def if_done_fn(_):
#             next_state = state
#             reward = jnp.float32(0)
#             done = state.done
#             next_obs = self.get_obs(state, params)
#             info = {
#                 "steps": state.steps,
#                 "reward": reward,
#                 "done_by_terminal": False,
#                 "done_by_horizon": False,
#                 "done_by_reward": False,
#             }
#             return next_state, reward, done, next_obs, info
#
#         # If not done, do the usual MDP transition
#         def if_not_done_fn(_):
#             next_state_idx = self.transitions[state.state_idx, action]
#             reward = self.rewards[state.state_idx, action]
#
#             new_steps = state.steps + 1
#             done_by_terminal = (next_state_idx == self.TERMINAL_STATE)
#             done_by_horizon = (new_steps >= params["horizon"])
#             done_by_reward = (reward != 0) & (params["done_on_reward"])
#
#             done_new = done_by_terminal | done_by_horizon | done_by_reward
#
#             reward += jnp.where(
#                 done_by_horizon & ~done_by_terminal,
#                 jnp.float32(params["no_done_reward"]),
#                 jnp.float32(0)
#             )
#
#             next_state = TabularState(
#                 state_idx=jnp.where(done_new, state.state_idx, next_state_idx),
#                 steps=new_steps,
#                 done=done_new
#             )
#             next_obs = self.get_obs(next_state, params)
#             info = {
#                 "steps": new_steps,
#                 "reward": reward,
#                 "done_by_terminal": done_by_terminal,
#                 "done_by_horizon": done_by_horizon,
#                 "done_by_reward": done_by_reward,
#             }
#             return next_state, reward, done_new, next_obs, info
#
#         return jax.lax.cond(
#             state.done,
#             if_done_fn,
#             if_not_done_fn,
#             operand=None
#         )
#
#     def get_obs(self, state: TabularState, params: Dict[str, Any]) -> jnp.ndarray:
#         """Get observation from state"""
#         if params["use_screen_observations"] and (self.screens is not None):
#             def valid_screen_fn(idx):
#                 return self.screens[self.screen_mapping[idx]]
#
#             def invalid_screen_fn(_):
#                 return jnp.zeros(self.screens.shape[1:], dtype=jnp.uint8)
#
#             return jax.lax.cond(
#                 (state.state_idx >= 0) & (state.state_idx < self.num_states),
#                 valid_screen_fn,
#                 invalid_screen_fn,
#                 state.state_idx
#             )
#         else:
#             return jnp.array(state.state_idx, dtype=jnp.int32)
#
#
# def create_tabular_env(problem_file: str) -> TabularEnv:
#     """Create a TabularEnv from a given .npz file path."""
#     return TabularEnv(problem_file)