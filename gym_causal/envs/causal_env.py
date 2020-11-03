import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import scipy.stats as  st
import random

class CausalEnv(gym.Env):
    """
    Description:
        A new coach has to decide whether a particuliar player must join
        the team during the training in order to decide if he is going to
        play the final match.


    Source:
        This task corresponds to the VOlleyball Task of Causal Project.


    Observation:
        Type: np.array
        Num     Observation
        0       The game is a loss
        1       The game is a win


    Actions:
        Type: np.array
        Num     Actions
        0       The player is not on the field
        1       The player is on the field

    Reward:
        The reward is 0 for the first 10 steps, then 1 if the game is a win,
        O otherwise.
        Alternatively, the reward is 0.1 for a win in the first 9 games,
        then 1 for a win in the final game, 0 otherwise.

    Starting State:
        The starting states is drawn from a uniform distribution over the two
        possible states.

    Episode Termination:
        After 11 time steps.

    """
    metadata = {'render.modes': ['human']}

    def __init__(self, p=0.8, q=0.2, reward_space=np.array([0, 0.1, 1])):
        self.observation_space = np.array([0, 1])
        self.action_space = np.array([0, 1])
        self.reward_space = reward_space
        self.time = 0
        self.probs = np.array([p, q])
        self.seed()
        self.viewer = None
        self.state = random.choice(self.observation_space)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert (action in self.action_space), err_msg

        done = (self.time == 10)

        # Compute the state:
        #     If the action is play, draw state from Bernoulli(p),
        #     else draw state from Bernoulli(q)
        if (action == 0):
          self.state = st.bernoulli.rvs(self.probs[1])
        elif (action == 1):
          self.state = st.bernoulli.rvs(self.probs[0])

        # Compute the reward:
        #     In the final game, if it's a win, reward is one, else it's zero.
        if (self.time == 10):
          if (self.state == 1):
              reward = self.reward_space[1]
          elif (self.state == 0):
              reward = self.reward_space[0]
        else:
            if (self.state == 1):
                reward = self.reward_space[0]
            elif (self.state == 0):
                reward = self.reward_space[0]

        self.time += 1

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.time = 0
        self.state = random.choice(self.observation_space)

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
          self.viewer.close()
          self.viewer = None

    def set_probs(self, p, q):
        self.probs = np.array([p, q])

    def set_rewards(self, rewards):
        self.reward_space = rewards
