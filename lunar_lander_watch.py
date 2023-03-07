import random
import os
from tqdm import tqdm
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

from LunarLanderEnvironment import LunarLanderEnvironment

class DQN():
    def __init__(self, numInputs, numOutputs, loadModel=None):
        self.numInputs = numInputs
        self.numOutputs = numOutputs
        self.replay_memory = []
        self.model = None
        self.targetModel = None
        if loadModel:
            self.model = models.load_model(loadModel)
            self.targetModel = self.createModel()
            self.targetModel.set_weights(self.model.get_weights())
        else:
            self.model = self.createModel()
            self.targetModel = self.createModel()
            self.targetModel.set_weights(self.model.get_weights())

    def createModel(self):
        model = models.Sequential(
            [layers.Dense(24, input_dim=self.numInputs, activation='tanh'),
            layers.Dense(48, activation='tanh'),
            layers.Dense(self.numOutputs, activation='linear')]
        )
        model.compile(loss='mse', optimizer=optimizers.Adam(lr=ALPHA), metrics=['accuracy'])
        return model

    def remember(self, transition):
        if len(self.replay_memory) > REPLAY_MEMORY_SIZE:
            self.replay_memory.pop(0)
        self.replay_memory.append(transition)
    
    def selectAction(self, state, epsilon):
        return random.randrange(env.actionSpaceSize) if (np.random.random() <= epsilon) else np.argmax(self.model.predict(np.array([state]))[0])

    def updateTarget(self):
        self.targetModel.set_weights(self.model.get_weights())

    def train(self):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.replay_memory, min(len(self.replay_memory), REPLAY_BATCH_SIZE))
        for transition in minibatch:
            state, action, reward, nextState, done = transition
            qValues = self.model.predict(np.array([state]))[0]
            qValues[action] = reward if done else reward + GAMMA * np.max(self.targetModel.predict(np.array([nextState]))[0])
            x_batch.append(state)
            y_batch.append(qValues)
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(minibatch), verbose=0)

ALPHA = 0.005
ALPHA_DECAY = 0.01
GAMMA = 0.99
EPSILON_MAX = 1.0
EPSILON_MIN = 0.01
EPSILON_LOG_DECAY = 0.01
epsilon = EPSILON_MAX
REPLAY_MEMORY_SIZE = 10_000
REPLAY_BATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5

numEpisodes = 1
numSteps = 600
verbose = True
if not os.path.isdir('lunar_lander_models'):
    os.makedirs('lunar_lander_models')
MODEL_NAME = "EPICstateSpace=desAccXY,angle,angVel_actionSpace=left,right,rear_rewardSpace=desAccRT40,angVelRT0.5_notes=sideThrusters(+-4.6,0.3),desVelMaxMag44-avg72.91057961820141avg" ## SETTING
checkPoint = "./lunar_lander_models/" + MODEL_NAME + ".model"

# For more repetitive results
# random.seed(1)
# np.random.seed(1)
# tf.set_random_seed(1)

epRewards = []

env = LunarLanderEnvironment()
agent = DQN(env.stateSpaceSize, env.actionSpaceSize, loadModel=checkPoint)

for episode in range(1, numEpisodes+1):
    if verbose: print(f"\nTesting on episode {episode}...")
    nextState = env.reset(test=True)
    env.renderInit()

    step = 0
    epReward = 0
    done = False

    while not done:
        state = nextState
        action = agent.selectAction(state, 0)
        nextState, reward, done, info = env.step(action, step, numSteps)
        step += 1
        epReward += reward

        report = f"State: {state[0]: >8.2f}, {state[1]: >8.2f}   Action: {action: >8.2f}   Reward: {reward: >8.2f}"
        print(report)
        env.render(report)
        
        if done:
            print(f"Abort status: {info['abortStatus']}")
            break
    avgReward = epReward / step
    if verbose: print(f"Testing on episode {episode} finished after {step} steps, avgReward: {avgReward}")
    env.closeRender()