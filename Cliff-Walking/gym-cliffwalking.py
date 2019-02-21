import gym
import random
import numpy as np

EPISODES=500
GAMMA=0.99
ALPHA=0.4
EPSILON_MIN = 0
EPSILON_DECAY = 0.5
EPSILON = 1

env = gym.make('CliffWalking-v0')
model = np.zeros([env.observation_space.n, env.action_space.n])
for x in range(env.observation_space.n):
        for action in range(env.action_space.n):
                model[x, action] = 0
print(model)
epi_reward = np.zeros([EPISODES])
epi_reward_average = np.zeros([EPISODES])
env.reset()
for episode in range(EPISODES):
    state = env.reset()
    done = False
    if np.random.rand() <= EPSILON:
        action = random.randrange(4)
    else:
        action = np.argmax(model[state, :])

    while not done:
        #env.render()
        new_state, R, done,_ = env.step(action)
        if np.random.rand() <= EPSILON:
            new_action = random.randrange(4)
        else:
            new_action = np.argmax(model[new_state, :])
        model[state][action]+=ALPHA*(R+GAMMA*np.amax(model[new_state][:])-model[state][action])
        state=new_state
        action=new_action
        epi_reward[episode] += R

    EPSILON*=EPSILON_DECAY
    if EPSILON<EPSILON_MIN:
        EPSILON=EPSILON_MIN
    if episode>0:
        epi_reward_average[episode] = np.mean(epi_reward[max(episode - 20, 0):episode])

    print('Episodio: ',episode,' Media Reward: ',epi_reward_average[episode],' Reward: ',epi_reward[episode], 'aleatoreidad: ', EPSILON)
print(model[24])