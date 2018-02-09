"""
A simple simulation of reaction times, generated as 
two distributions of lognormal values (with a slight right
skewness), with significant overlap between the two distributions.
The non-action version is set to have values with mean of ~0.35, 
max ~1.55, min ~0.07. The action version (simulating
a version with some kind of feedback, say a phasic alert, that improves RT)
has mean of ~0.26, min ~ 0.045, max ~ 1.12. Which action to take
is learned almost perfectly (conditional on the exploration strategy parameters)
with DQN and SARSA. With many sets of params, for a shallow network with
a small probability of exploration being done, both learn about equally
fast (within a few time steps). For deeper networks, DQN learns at 
least a few times faster, unsuprisingly.
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding

#for generating the simulated RTs
import numpy as np
from random import uniform
import matplotlib
matplotlib.use('wxagg')
import matplotlib.pyplot as plt
from pylab import *



logger = logging.getLogger(__name__)

class RTSimulationEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def softmax(self,x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def __init__(self):
        self.windowSize = 10 #used for reward calculation
        self.action_space = spaces.Discrete(2) #how many actions can we feedback?
        self.observation_space = spaces.Box(low=0, high=2, shape=(6,1))
        self.totalSteps = 200
        self.states = []
        self.totalStates = []
        self.actions = []
        self.rewards = []
        self.probabilityOfFeedback = 1
        self.energies = []
        self._seed()
        self.viewer = None
        self.state = None
        self.testing = False
        self.steps_beyond_done = None

    def _seed(self, seed=1729):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        #update the current state, based on the action taken (non-alert vs alert)
        if(action == 0):
            self.state = self.np_random.lognormal(-1.09, 0.175, 1)
            self.delta = self.np_random.lognormal(-1.20, 0.286, 1)
            self.theta = self.np_random.lognormal(-1.67, 0.304, 1)
            self.alpha = self.np_random.lognormal(-1.78, 0.356, 1)
            self.beta = self.np_random.lognormal(-1.69, 0.271, 1)
            self.gamma = self.np_random.lognormal(-2.21, 0.312, 1)
        elif(action == 1):
            if(np.random.uniform(0,1) <= self.probabilityOfFeedback):
                self.state = self.np_random.lognormal(-1.17, 0.174, 1)
                self.delta = self.np_random.lognormal(-1.300,0.300,1)
                self.theta = self.np_random.lognormal(-1.67,0.272,1)
                self.alpha = self.np_random.lognormal(-1.59,0.365,1)
                self.beta = self.np_random.lognormal(-1.69,0.228,1)
                self.gamma = self.np_random.lognormal(-2.22,0.336,1)
            else:
                self.state = self.np_random.lognormal(-1.09, 0.175, 1)
                self.delta = self.np_random.lognormal(-1.20, 0.286, 1)
                self.theta = self.np_random.lognormal(-1.67, 0.304, 1)
                self.alpha = self.np_random.lognormal(-1.78, 0.356, 1)
                self.beta = self.np_random.lognormal(-1.69, 0.271, 1)
                self.gamma = self.np_random.lognormal(-2.21, 0.312, 1)
                
        
        softmaxedBandPowers = self.softmax([self.delta, self.theta, self.alpha, self.beta, self.gamma])
        self.delta = softmaxedBandPowers[0]
        self.theta = softmaxedBandPowers[1]
        self.alpha = softmaxedBandPowers[2]
        self.beta = softmaxedBandPowers[3]
        self.gamma = softmaxedBandPowers[4]
        
        #self.energies += [self.energy]
        self.actions += [action]
        self.states += [self.state]
        self.totalStates += [self.state]
        done = len(self.states) > self.totalSteps - 1

        if not done:
            if len(self.states) > self.windowSize:
                #simplest rewards are the easiest to tune
#                if np.mean(self.states[-(self.windowSize+1):-1]) >= (self.states[-1]):
#                    reward = 1
#                else:
#                    reward = -1
                #other alternatives I've tried:
                [reward] = np.mean(self.states[-(self.windowSize+1):-1]) - self.states[-1]
            else:
                reward = 0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 0

            #previously:
            #[reward] = np.mean(self.states[-(self.windowSize+1):-1]) - self.states[-1]
        else:



            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        self.rewards += [reward]
        return np.array([self.state, self.delta, self.theta, self.alpha, self.beta, self.gamma]), reward, done, {}

    def _reset(self):

        # We just finished simulating RTs
        # Show a graph of simulated and learned results
        print "----------------Resetting----------------------"

        self.state = self.np_random.lognormal(-1.09, 0.175, 1)
        self.delta = self.np_random.lognormal(-1.20, 0.286, 1)
        self.theta = self.np_random.lognormal(-1.67, 0.304, 1)
        self.alpha = self.np_random.lognormal(-1.78, 0.356, 1)
        self.beta = self.np_random.lognormal(-1.69, 0.271, 1)
        self.gamma = self.np_random.lognormal(-2.21, 0.312, 1)
        self.states = []
        self.steps_beyond_done = None
        return np.array([self.state, self.delta, self.theta, self.alpha, self.beta, self.gamma])
#        return np.array([self.state])

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)


        if self.state is None: return None

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
