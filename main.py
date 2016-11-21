import gym
import json
from random import random, randint

slow = True
env = gym.make("MountainCar-v0")
VERSION = "V4"
measurements = []
exploration_rate = 0.0
dynamic_exploration = False
episodes = 100
qvalues = {}
last_o = None
last_a = None
path = []


def take_action(t, o, r):
    global last_a, last_o, qvalues
    act_back, act_none, act_break = range(3)
    best_a = get_best_action(o)
    a = randint(0,2) if random() < exploration_rate else best_a
    idx = get_index(o, a)
    if last_o is not None:
        best_q = qvalues[idx][0] if idx in qvalues else -1  # initialize all qvalues
        qvalues[get_index(last_o, last_a)] = (r + best_q, idx_o(o))
    last_o, last_a = o, a
    path.append(idx)
    return a


def update_done():
    qvalues[get_index(last_o, last_a)] = (0, None)  # set the last reward
    for n, i in enumerate(path[1::-1]):
        o = qvalues[i][1][1:-1]
        best_a = get_best_action(o, True)
        best_q = qvalues[get_index([int(j) for j in o.split(',')], last_a, True)]
        update = (qvalues[i][0] + 0.8**n*(-1 + best_q[0]))/2
        qvalues[i] = (update, '('+o+')')


def get_best_action(o, is_index=False):
    values = []
    for a in range(3):
        idx = get_index(o, a) if not is_index else o
        values.append(qvalues[idx] if idx in qvalues else (0, None))
    return values.index(max(values,key=lambda item: item[0]))


def get_index(o, a, no_transform=False):
    return idx_o(o, no_transform)+str(a)


def idx_o(o, no_transform=False):
    if no_transform:
        return str((o[0], o[1]))
    else:
        return str((int(o[0] * 100), int(o[1] * 100)))

for i in range(episodes):
    observation = env.reset()
    reward = 0
    done = False
    timesteps = 0
    if dynamic_exploration: exploration_rate = episodes / (2*(episodes-i))
    while not done:
        if slow and i == episodes-1: env.render()
        action = take_action(timesteps, observation, reward)
        observation, reward, done, info = env.step(action)
        timesteps += 1
    update_done()
    print ("Episode {} finished after {} timesteps.".format(i, timesteps))
    measurements.append(timesteps)

avg_timesteps = sum(measurements)/len(measurements)
#print(json.dumps(qvalues, indent=4, sort_keys=True))
print("All {} episodes took {} on average".format(episodes, avg_timesteps))

if not slow:
    with open("results.txt", "a+") as fout:
        fout.write("{} Exploration Rate {} AVG {} {}\n".format(VERSION, exploration_rate, avg_timesteps, measurements))