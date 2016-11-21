import gym
from random import randint

slow = False
env = gym.make("MountainCar-v0")
VERSION = "V1"
measurements = []
episodes = 10
qvalues = {}
last_observation = None


def take_action(t, o, r):
    act_back, act_none, act_break = range(3)
    exploration_rate = 0.0
    best_a = get_best_action(o)
    a = randint(0,3) if randint(0,100) < exploration_rate*100 else best_a
    if last_observation is not None:
        idx = get_index(o, best_a)
        bestq = qvalues[idx] if idx in qvalues else 0
        qvalues[get_index(o, a)] = r + bestq
    return env.action_space.sample()


def get_best_action(o):
    values = []
    for a in range(3):
        idx = get_index(o, a)
        values.append(qvalues[idx] if idx in qvalues else 0)  # initialize all qvalues to 0
    return values.index(max(values))


def get_index(o, a):
    new_o = int(o[0] * 100), int(o[1] * 100)
    return str(new_o)+str(a)

for _ in range(episodes):
    observation = env.reset()
    reward = 0
    done = False
    timesteps = 0
    while not done:
        if slow: env.render()
        action = take_action(timesteps, observation, reward)
        observation, reward, done, info = env.step(action)
        timesteps+=1
        if slow: print (observation)
        if slow: print (reward)
        if slow: print (done)
    print ("Episode finished after ", timesteps, "timesteps.")
    measurements.append(timesteps)

avg_timesteps = sum(measurements)/len(measurements)
print("All {} episodes took {} on average".format(episodes, avg_timesteps))

if not slow:
    with open("results.txt", "a+") as fout:
        fout.write("{} {} AVG {}\n".format(VERSION, measurements, avg_timesteps))