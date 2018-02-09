import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.agents.sarsa import SarsaAgent
from rl.agents.ddpg import DDPGAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, GreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

#for saving simulation results for later analysis/prettier graphing in MATLAB
import csv
from datetime import datetime
import calendar

def dqn_rt(nonLinearLayers=3, neuronsPerLayer = 4, epsilon = 0.3, tau = 1,
           exploration="tau", gamma = 0.5):
    
    ENV_NAME = 'RtSimulationEnv-v0'
    
    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n
    
    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    for x in range(0, nonLinearLayers):
        model.add(Activation('relu'))
        model.add(Dense(neuronsPerLayer))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())
    
    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=50000, window_length=1)
    if(exploration == "epsilon"):
        policy = EpsGreedyQPolicy(eps=epsilon)
    else:
        policy = BoltzmannQPolicy(tau=tau)

    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,target_model_update=1e-2, policy=policy, gamma = gamma)
    #dqn = SarsaAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=10, policy=policy)
    #dqn = DDPGAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    
    # Okay, now it's time to learn something! If we visualize the training here for show, this
    # slows down training quite a lot. We can also always safely abort the training prematurely using
    # Ctrl + C.
    env.testing = False
    foo = dqn.fit(env, nb_steps=1000, visualize=False, verbose=3)

    d = datetime.utcnow()
    unixtime = calendar.timegm(d.utctimetuple())
    ###save env.unwrapped.totalStates and env.unwrapped.actions as: [state,action] pairs
    with open('simulatedRTs_' + str(unixtime) + '.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows([env.unwrapped.totalStates, env.unwrapped.rewards, env.unwrapped.actions])
    
    # After training is done, we save the final weights.
    # dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
    
    # Finally, evaluate our algorithm for 10 episodes.
    env.testing = True
    dqn.test(env, nb_episodes=5, visualize=False)

dqn_rt()
