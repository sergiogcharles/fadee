"""Simple grid world environments."""
import itertools

import gym
import numpy as np

from envs import grid
import meta_exploration


class InstructionWrapper(meta_exploration.InstructionWrapper):
    """Instruction wrapper for CityGridEnv.

    Provides instructions (goal locations) and their corresponding rewards.

    Reward function for a given goal is:
            R(s, a) = -0.3 if s != goal
                    = 1      otherwise
    """
    GOALS = [np.array((0, 0)), np.array((4, 4)),
             np.array((0, 4)), np.array((4, 0))] #, np.array((4, 0)), np.array((4, 8))]
    # GOALS = [np.array((0, 0)), np.array((8, 8)),
    #           np.array((0, 8)), np.array((8, 0))]
    

    def _instruction_observation_space(self):
        return gym.spaces.Box(
                np.array([0, 0]), np.array([self.width, self.height]),
                dtype=np.int)

    def _reward(self, instruction_state, action, original_reward):
        del original_reward

        done = False
        reward = -0.3
        if np.array_equal(self.agent_pos, instruction_state.instructions):
            reward = 1
        elif action == grid.Action.end_episode:
            reward -= self.steps_remaining * 0.3  # penalize ending the episode

        done = any(np.array_equal(self.agent_pos, goal) for goal in self.GOALS)
        return reward, done

    def _generate_instructions(self, test=False):
        del test

        #goal = self.GOALS[self._random.randint(len(self.GOALS))]


        if self.env.env_id[0] >= 24:
            goal = self.GOALS[0]
        else:
            goal = self.GOALS[self._random.randint(len(self.GOALS))]
        return goal

    def render(self, mode="human"):
        image = super().render(mode)
        image.draw_rectangle(self.current_instructions, 0.5, "green")
        image.write_text(f"Instructions: {self.current_instructions}")
        return image

    def __str__(self):
        s = super().__str__()
        s += f"\nInstructions: {self.current_instructions}"
        return s


class CityGridEnv(grid.GridEnv):
    """Defines a city grid with bus stops at fixed locations.

    Upon toggling a bus stop, the agent is teleported to the next bus stop.
    - The environment defines no reward function (see InstructionWrapper for
    rewards).
    - The episode ends after a fixed number of steps.
    - Different env_ids correspond to different bus destination permutations.
    """

    # Location of the bus stops and the color to render them
    _bus_sources = [
            (np.array((2, 1)), "rgb(0,0,255)"),
            (np.array((3, 2)), "rgb(255,0,255)"),
            (np.array((2, 3)), "rgb(255,255,0)"),
            (np.array((1, 2)), "rgb(0,255,255)"),
    ]

    _destinations = [
            np.array((0, 0)), np.array((0, 4)),
            np.array((4, 4)), np.array((4, 0)),
    ]

    _bus_permutations = list(itertools.permutations(_destinations))

    _height = 5
    _width = 5

    # Optimization: Full set of train / test IDs is large, so only compute it
    # once. Even though individual IDs are small, the whole ID matrix cannot be
    # freed if we have a reference to a single ID.
    _train_ids = None
    _test_ids = None

    def __init__(self, env_id, wrapper, max_steps=10):
        super().__init__(env_id, wrapper, max_steps=max_steps,
                         width=self._width, height=self._height)

    @classmethod
    def instruction_wrapper(cls):
        return InstructionWrapper

    def _env_id_space(self):
        low = np.array([0])
        high = np.array([len(self._bus_permutations)])
        dtype = np.int
        return low, high, dtype

    @classmethod
    def env_ids(cls):
        ids = np.expand_dims(np.array(range(len(cls._bus_permutations))), 1)
        return np.array(ids), np.array(ids)

    def text_description(self):
        return "bus grid"

    def _place_objects(self):
        super()._place_objects()
        self._agent_pos = np.array([2, 2])

        destinations = self._bus_permutations[
                self.env_id[0] % len(self._bus_permutations)]
        for (bus_stop, color), dest in zip(self._bus_sources, destinations):
            self.place(grid.Bus(color, dest), bus_stop)
            self.place(grid.Bus(color, bus_stop), dest)


class MapGridEnv(CityGridEnv):
    """Includes a map that tells the bus orientations."""

    def _observation_space(self):
        low, high, dtype = super()._observation_space()
        # add dim for map
        env_id_low, env_id_high, _ = self._env_id_space()

        low = np.concatenate((low, [env_id_low[0]]))
        high = np.concatenate((high, [env_id_high[0] + 1]))
        return low, high, dtype

    def text_description(self):
        return "map grid"

    def _place_objects(self):
        super()._place_objects()
        self._map_pos = np.array([4, 2])

    def _gen_obs(self):
        obs = super()._gen_obs()
        map_info = [0]
        if np.array_equal(self.agent_pos, self._map_pos):
            map_info = [self.env_id[0] + 1]
        return np.concatenate((obs, map_info), 0)

    def render(self, mode="human"):
        image = super().render(mode=mode)
        image.draw_rectangle(self._map_pos, 0.4, "black")
        return image

class LargeCityGridEnv(grid.GridEnv):
    """Defines a city grid with bus stops at fixed locations.

    Upon toggling a bus stop, the agent is teleported to the next bus stop.
    - The environment defines no reward function (see InstructionWrapper for
    rewards).
    - The episode ends after a fixed number of steps.
    - Different env_ids correspond to different bus destination permutations.
    """

    # Location of the bus stops and the color to render them
    _bus_sources = [
            (np.array((4, 3)), "rgb(0,0,255)"),
            (np.array((5, 4)), "rgb(255,0,255)"),
            (np.array((4, 5)), "rgb(255,255,0)"),
            (np.array((3, 4)), "rgb(0,255,255)"),
            (np.array((3, 3)), "rgb(0,255,0)"),
            (np.array((5, 3)), "rgb(0,0,255)")
    ]

    _destinations = [
            np.array((0, 0)), np.array((0, 8)),
            np.array((8, 8)), np.array((8, 0)), np.array((4, 0)), np.array((4, 8)),
    ]

    _bus_permutations = list(itertools.permutations(_destinations))

    _height = 9
    _width = 9

    # Optimization: Full set of train / test IDs is large, so only compute it
    # once. Even though individual IDs are small, the whole ID matrix cannot be
    # freed if we have a reference to a single ID.
    _train_ids = None
    _test_ids = None

    def __init__(self, env_id, wrapper, max_steps=10):
        super().__init__(env_id, wrapper, max_steps=max_steps,
                         width=self._width, height=self._height)

    @classmethod
    def instruction_wrapper(cls):
        return InstructionWrapper

    def _env_id_space(self):
        low = np.array([0])
        high = np.array([len(self._bus_permutations)])
        dtype = np.int
        return low, high, dtype

    @classmethod
    def env_ids(cls):
        ids = np.expand_dims(np.array(range(len(cls._bus_permutations))), 1)
        return np.array(ids), np.array(ids)

    def text_description(self):
        return "bus grid"

    def _place_objects(self):
        super()._place_objects()
        self._agent_pos = np.array([4, 4])

        destinations = self._bus_permutations[
                self.env_id[0] % len(self._bus_permutations)]
        for (bus_stop, color), dest in zip(self._bus_sources, destinations):
            self.place(grid.Bus(color, dest), bus_stop)
            self.place(grid.Bus(color, bus_stop), dest)
class LargeMapGridEnv(LargeCityGridEnv):
    """Includes a map that tells the bus orientations."""

    def _observation_space(self):
        low, high, dtype = super()._observation_space()
        # add dim for map
        env_id_low, env_id_high, _ = self._env_id_space()

        low = np.concatenate((low, [env_id_low[0]]))
        high = np.concatenate((high, [env_id_high[0] + 1]))
        return low, high, dtype

    def text_description(self):
        return "map grid"

    def _place_objects(self):
        super()._place_objects()
        self._map_pos = np.array([6, 4])

    def _gen_obs(self):
        obs = super()._gen_obs()
        map_info = [0]
        if np.array_equal(self.agent_pos, self._map_pos):
            map_info = [self.env_id[0] + 1]
        return np.concatenate((obs, map_info), 0)

    def render(self, mode="human"):
        image = super().render(mode=mode)
        image.draw_rectangle(self._map_pos, 0.4, "black")
        return image

class DoubleMapGridEnv(CityGridEnv):
    """Includes a map that tells the bus orientations."""

    def _observation_space(self):
        low, high, dtype = super()._observation_space()
        # add dim for map
        env_id_low, env_id_high, _ = self._env_id_space()

        low = np.concatenate((low, [env_id_low[0]]))
        high = np.concatenate((high, [env_id_high[0] + 1]))
        return low, high, dtype

    def text_description(self):
        return "map grid double"

    def _place_objects(self):
        super()._place_objects()
        self._map_pos = np.array([4, 2])

    def _gen_obs(self):
        obs = super()._gen_obs()
        map_info = [0]
        if np.array_equal(self.agent_pos, self._map_pos):
            map_info = [self.env_id[0] + 1]
        return np.concatenate((obs, map_info), 0)

    def render(self, mode="human"):
        image = super().render(mode=mode)
        image.draw_rectangle(self._map_pos, 0.4, "black")
        return image
class UpdatedGridEnv(grid.GridEnv):
    """Defines a city grid with bus stops at fixed locations.

    Upon toggling a bus stop, the agent is teleported to the next bus stop.
    - The environment defines no reward function (see InstructionWrapper for
    rewards).
    - The episode ends after a fixed number of steps.
    - Different env_ids correspond to different bus destination permutations.
    """

    # Location of the bus stops and the color to render them
    _bus_sources = [
            (np.array((2, 1)), "rgb(0,0,255)"),
            (np.array((3, 2)), "rgb(255,0,255)"),
            (np.array((2, 3)), "rgb(255,255,0)"),
            (np.array((1, 2)), "rgb(0,255,255)"),
    ]

    _teleportation_sources = [
        (np.array((1, 1)), "rgb(0,0,0)"),
        (np.array((1, 2)), "rgb(0,0,0)"),
        (np.array((1, 3)), "rgb(0,0,0)"),
        (np.array((2, 3)), "rgb(0,0,0)"),
        (np.array((3, 3)), "rgb(0,0,0)"),
        (np.array((3, 2)), "rgb(0,0,0)"),
        (np.array((3, 1)), "rgb(0,0,0)"),
        (np.array((2, 1)), "rgb(0,0,0)"),
    ]

    _destinations = [
            np.array((0, 0)), np.array((0, 4)),
            np.array((4, 4)), np.array((4, 0)),
    ]

    _teleportation_destination = np.array((0, 0))

    _bus_permutations = list(itertools.permutations(_destinations))



    _height = 5
    _width = 5

    # Optimization: Full set of train / test IDs is large, so only compute it
    # once. Even though individual IDs are small, the whole ID matrix cannot be
    # freed if we have a reference to a single ID.
    _train_ids = None
    _test_ids = None

    def __init__(self, env_id, wrapper, max_steps=10):
        super().__init__(env_id, wrapper, max_steps=max_steps,
                         width=self._width, height=self._height)

    @classmethod
    def instruction_wrapper(cls):
        return InstructionWrapper

    def _env_id_space(self):
        low = np.array([0])
        high = np.array([len(self._bus_permutations) + 8])
        dtype = np.int
        return low, high, dtype

    @classmethod
    def env_ids(cls):
        ids = np.expand_dims(np.array(range(len(cls._bus_permutations) + 8)), 1)
        return np.array(ids), np.array(ids)

    def text_description(self):
        return "bus grid"
    
    def _observation_space(self):
        low, high, dtype = super()._observation_space()
        # add dim for map
        env_id_low, env_id_high, _ = self._env_id_space()

        low = np.concatenate((low, [env_id_low[0]]))
        high = np.concatenate((high, [env_id_high[0] + 1]))
        return low, high, dtype
    def _gen_obs(self):
        obs = super()._gen_obs()
        map_info = [0]
        if self.env_id[0] < 24:
            if np.array_equal(self.agent_pos, self._map_pos):
                map_info = [self.env_id[0] + 1]  
        return np.concatenate((obs, map_info), 0)

    def _place_objects(self):
        super()._place_objects()
        self._agent_pos = np.array([2, 2])

        if self.env_id[0] < 24:
            self._map_pos = np.array([4, 2])
            destinations = self._bus_permutations[
                    self.env_id[0] % len(self._bus_permutations)]
            for (bus_stop, color), dest in zip(self._bus_sources, destinations):
                self.place(grid.Bus(color, dest), bus_stop)
                self.place(grid.Bus(color, bus_stop), dest)
        else:
            pair = self._teleportation_sources[self.env_id[0] - 24]
            self.place(grid.Bus(pair[1], self._teleportation_destination), pair[0])
    def render(self, mode="human"):
        image = super().render(mode=mode)
        if self.env_id[0] < 24:
            image.draw_rectangle(self._map_pos, 0.4, "black")
        return image
class UpdatedTwoGridEnv(grid.GridEnv):
    """Defines a city grid with bus stops at fixed locations.

    Upon toggling a bus stop, the agent is teleported to the next bus stop.
    - The environment defines no reward function (see InstructionWrapper for
    rewards).
    - The episode ends after a fixed number of steps.
    - Different env_ids correspond to different bus destination permutations.
    """

    # Location of the bus stops and the color to render them
    _bus_sources = [
            (np.array((4, 3)), "rgb(0,0,255)"),
            (np.array((5, 4)), "rgb(255,0,255)"),
            (np.array((4, 5)), "rgb(255,255,0)"),
            (np.array((3, 4)), "rgb(0,255,255)"),
    ]

    _teleportation_sources = [
        (np.array((3, 3)), "rgb(0,0,0)"),
        (np.array((3, 4)), "rgb(0,0,0)"),
        (np.array((3, 5)), "rgb(0,0,0)"),
        (np.array((4, 5)), "rgb(0,0,0)"),
        (np.array((5, 5)), "rgb(0,0,0)"),
        (np.array((5, 4)), "rgb(0,0,0)"),
        (np.array((5, 3)), "rgb(0,0,0)"),
        (np.array((4, 3)), "rgb(0,0,0)"),
    ]

    _teleportation_sources_second = [
        (np.array((3, 2)), "rgb(0,0,0)"),
        (np.array((2, 2)), "rgb(0,0,0)"),
        (np.array((2, 3)), "rgb(0,0,0)"),
        (np.array((2, 4)), "rgb(0,0,0)"),
        (np.array((2, 5)), "rgb(0,0,0)"),
        (np.array((2, 6)), "rgb(0,0,0)"),
        (np.array((3, 6)), "rgb(0,0,0)"),
        (np.array((4, 6)), "rgb(0,0,0)"),
    ]

    _destinations = [
            np.array((0, 1)), np.array((0, 7)), np.array((8, 1)), np.array((8, 7)),
    ]

    _teleportation_destination = np.array((0, 0))

    _bus_permutations = list(itertools.permutations(_destinations))



    _height = 9
    _width = 9

    # Optimization: Full set of train / test IDs is large, so only compute it
    # once. Even though individual IDs are small, the whole ID matrix cannot be
    # freed if we have a reference to a single ID.
    _train_ids = None
    _test_ids = None

    def __init__(self, env_id, wrapper, max_steps=10):
        super().__init__(env_id, wrapper, max_steps=max_steps,
                         width=self._width, height=self._height)

    @classmethod
    def instruction_wrapper(cls):
        return InstructionWrapper

    def _env_id_space(self):
        low = np.array([0])
        high = np.array([len(self._bus_permutations) + 16])
        dtype = np.int
        return low, high, dtype

    @classmethod
    def env_ids(cls):
        ids = np.expand_dims(np.array(range(len(cls._bus_permutations) + 16)), 1)
        return np.array(ids), np.array(ids)

    def text_description(self):
        return "bus grid"
    
    def _observation_space(self):
        low, high, dtype = super()._observation_space()
        # add dim for map
        env_id_low, env_id_high, _ = self._env_id_space()

        low = np.concatenate((low, [env_id_low[0]], [0]))
        high = np.concatenate((high, [env_id_high[0] + 1], [3]))
        return low, high, dtype
    def _gen_obs(self):
        obs = super()._gen_obs()
        map_info = [0]
        sign_info = [0]
        if self.env_id[0] < 24:
            if np.array_equal(self.agent_pos, self._map_pos):
                map_info = [self.env_id[0] + 1]
        elif self.env_id[0] < 32:
            if np.array_equal(self.agent_pos, self._sign_pos):
                sign_info = [1]
        else:
            if np.array_equal(self.agent_pos, self._sign_pos):
                sign_info = [2]  
        return np.concatenate((obs, map_info, sign_info), 0)

    def _place_objects(self):
        super()._place_objects()
        self._agent_pos = np.array([4, 4])

        if self.env_id[0] < 24:
            self._map_pos = np.array([6, 4])
            destinations = self._bus_permutations[
                    self.env_id[0] % len(self._bus_permutations)]
            for (bus_stop, color), dest in zip(self._bus_sources, destinations):
                self.place(grid.Bus(color, dest), bus_stop)
                self.place(grid.Bus(color, bus_stop), dest)
        elif self.env_id[0] < 32:
            self._sign_pos = np.array([4, 2])
            pair = self._teleportation_sources[self.env_id[0] - 24]
            self.place(grid.Bus(pair[1], self._teleportation_destination), pair[0])
        else:
            self._sign_pos = np.array([4, 2])
            pair = self._teleportation_sources_second[self.env_id[0] - 32]
            self.place(grid.Bus(pair[1], self._teleportation_destination), pair[0])


    def render(self, mode="human"):
        image = super().render(mode=mode)
        if self.env_id[0] < 24:
            image.draw_rectangle(self._map_pos, 0.4, "black")
        else:
            image.draw_rectangle(self._sign_pos, 0.4, "black")
        return image

class UpdatedThreeGridEnv(grid.GridEnv):
    """Defines a city grid with bus stops at fixed locations.

    Upon toggling a bus stop, the agent is teleported to the next bus stop.
    - The environment defines no reward function (see InstructionWrapper for
    rewards).
    - The episode ends after a fixed number of steps.
    - Different env_ids correspond to different bus destination permutations.
    """

    # Location of the bus stops and the color to render them
    _bus_sources = [
            (np.array((4, 3)), "rgb(0,0,255)"),
            (np.array((5, 4)), "rgb(255,0,255)"),
            (np.array((4, 5)), "rgb(255,255,0)"),
            (np.array((3, 4)), "rgb(0,255,255)"),
    ]

    _teleportation_sources = [
        (np.array((3, 4)), "rgb(0,0,0)"),
        (np.array((2, 4)), "rgb(0,0,0)"),
        (np.array((3, 5)), "rgb(0,0,0)"),
        (np.array((2, 5)), "rgb(0,0,0)"),
    ]

    _teleportation_sources_second = [
        (np.array((5, 4)),  "rgb(0,0,0)"),
        (np.array((5, 5)), "rgb(0,0,0)"),
        (np.array((6, 4)), "rgb(0,0,0)"),
        (np.array((6, 5)), "rgb(0,0,0)"),
    ]

    _teleportation_sources_third = [
        (np.array((3, 2)), "rgb(0,0,0)"),
        (np.array((2, 2)), "rgb(0,0,0)"),
        (np.array((2, 3)), "rgb(0,0,0)"),
    ]

    _teleportation_sources_fourth = [
        (np.array((5, 2)), "rgb(0,0,0)"),
        (np.array((6, 2)), "rgb(0,0,0)"),
        (np.array((6, 3)), "rgb(0,0,0)"),
    ]

    _destinations = [
            np.array((0, 1)), np.array((0, 7)), np.array((8, 1)), np.array((8, 7)),
    ]

    _teleportation_destination = np.array((0, 0))

    _bus_permutations = list(itertools.permutations(_destinations))



    _height = 9
    _width = 9

    # Optimization: Full set of train / test IDs is large, so only compute it
    # once. Even though individual IDs are small, the whole ID matrix cannot be
    # freed if we have a reference to a single ID.
    _train_ids = None
    _test_ids = None

    def __init__(self, env_id, wrapper, max_steps=10):
        super().__init__(env_id, wrapper, max_steps=max_steps,
                         width=self._width, height=self._height)

    @classmethod
    def instruction_wrapper(cls):
        return InstructionWrapper

    def _env_id_space(self):
        low = np.array([0])
        high = np.array([len(self._bus_permutations) + 14])
        dtype = np.int
        return low, high, dtype

    @classmethod
    def env_ids(cls):
        ids = np.expand_dims(np.array(range(len(cls._bus_permutations) + 14)), 1)
        return np.array(ids), np.array(ids)

    def text_description(self):
        return "bus grid"
    
    def _observation_space(self):
        low, high, dtype = super()._observation_space()
        # add dim for map
        env_id_low, env_id_high, _ = self._env_id_space()

        low = np.concatenate((low, [env_id_low[0]], [0], [0]))
        high = np.concatenate((high, [env_id_high[0] + 1], [3], [3]))
        return low, high, dtype
    def _gen_obs(self):
        obs = super()._gen_obs()
        map_info = [0]
        sign_info = [0]
        task_info = [0]
        if self.env_id[0] < 24:
            if np.array_equal(self.agent_pos, self._map_pos):
                map_info = [self.env_id[0] + 1]
        elif self.env_id[0] < 28:
            if np.array_equal(self.agent_pos, self._sign_pos_one):
                sign_info = [2]
            if np.array_equal(self.agent_pos, self._sign_pos_two):
                task_info = [2]
        elif self.env_id[0] < 32:
            if np.array_equal(self.agent_pos, self._sign_pos_one):
                sign_info = [1]
            if np.array_equal(self.agent_pos, self._sign_pos_two):
                task_info = [2]
        elif self.env_id[0] < 35:
            if np.array_equal(self.agent_pos, self._sign_pos_one):
                sign_info = [2]
            if np.array_equal(self.agent_pos, self._sign_pos_two):
                task_info = [1]
        else:
            if np.array_equal(self.agent_pos, self._sign_pos_one):
                sign_info = [1]
            if np.array_equal(self.agent_pos, self._sign_pos_two):
                task_info = [1]
        return np.concatenate((obs, map_info, sign_info, task_info), 0)

    def _place_objects(self):
        super()._place_objects()
        self._agent_pos = np.array([4, 4])

        if self.env_id[0] < 24:
            self._map_pos = np.array([6, 4])
            destinations = self._bus_permutations[
                    self.env_id[0] % len(self._bus_permutations)]
            for (bus_stop, color), dest in zip(self._bus_sources, destinations):
                self.place(grid.Bus(color, dest), bus_stop)
                self.place(grid.Bus(color, bus_stop), dest)
        elif self.env_id[0] < 28:
            self._sign_pos_one = np.array([4, 5])
            self._sign_pos_two = np.array([3, 3])

            pair = self._teleportation_sources[self.env_id[0] - 24]
            self.place(grid.Bus(pair[1], self._teleportation_destination), pair[0])
        elif self.env_id[0] < 32:
            self._sign_pos_one = np.array([4, 5])
            self._sign_pos_two = np.array([5, 3])

            pair = self._teleportation_sources_second[self.env_id[0] - 28]
            self.place(grid.Bus(pair[1], self._teleportation_destination), pair[0])
        elif self.env_id[0] < 35:
            self._sign_pos_one = np.array([4, 5])
            self._sign_pos_two = np.array([3, 3])

            pair = self._teleportation_sources_third[self.env_id[0] - 32]
            self.place(grid.Bus(pair[1], self._teleportation_destination), pair[0])
        else:
            self._sign_pos_one = np.array([4, 5])
            self._sign_pos_two = np.array([5, 3])

            pair = self._teleportation_sources_fourth[self.env_id[0] - 35]
            self.place(grid.Bus(pair[1], self._teleportation_destination), pair[0])

    def render(self, mode="human"):
        image = super().render(mode=mode)
        if self.env_id[0] < 24:
            image.draw_rectangle(self._map_pos, 0.4, "black")
        else:
            image.draw_rectangle(self._sign_pos_one, 0.4, "black")
            image.draw_rectangle(self._sign_pos_two, 0.4, "black")
        return image
            
        