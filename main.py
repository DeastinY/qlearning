import sys
import gym
import operator
from random import random, randint

slow = True
env = gym.make("MountainCar-v0")
VERSION = "V1.6"
measurements = []
exploration_rate = 0.0
exploration_decay = 0.999
decay_rate = 1.0
dynamic_exploration = False
episodes = 300
all_qvalues = {}  # { '(position, speed)': [{'action': {'Action' : A, 'Qvalue': Q, 'ResultState': (position, speed)]}}
path = []
last_action = None
last_position = None
last_precise_position = None
last_speed = None
last_index = None


def take_action(observation, reward):
    # 0 = back, 1 = nothing, 2 = front
    global last_action, last_position, last_speed, last_index, last_precise_position, exploration_decay, exploration_rate
    precise_position = observation[0]
    position = discretize_position(precise_position)
    action, index, speed = None, None, None

    if last_action is None:
        action = randint(0, 2)
        speed = 0
        index = get_index(position, speed)
        all_qvalues[index] = [{'action' : action, 'qvalue' : reward, 'resulststate' : None}]
    else:
        speed = discretize_speed(calculate_speed(last_precise_position, precise_position))
        index = get_index(position, speed)
        last_index = get_index(last_position, last_speed)
        exploration_rate *= exploration_decay
        action = randint(0,2) if random() < exploration_rate else get_best_action(index)
        update_state(last_index, last_action, reward+best_qvalue(index), index)

    last_action, last_position, last_precise_position, last_index, last_speed = \
        action, position, precise_position, index, speed
    path.append((last_index, last_action))
    return action


def update_state(index, action, qvalue, resultstate):
    for a in all_qvalues[last_index]:
        if a['action'] == action:
            a['resultstate'] = resultstate
            a['qvalue'] = qvalue
            break
    else:
        all_qvalues[last_index].append({'action' : action, 'qvalue' : qvalue, 'resulststate' : resultstate})


def best_qvalue(index):
    max_q = None
    for a in all_qvalues[index]:
        q = a['qvalue']
        max_q = q if max_q is None or q > max_q else max_q
    return max_q if max_q is not None else 0


def calculate_speed(old_position, position):
    return old_position-position


def discretize_position(position):
    return int(position*100)


def discretize_speed(speed):
    return int(speed*1000)


def get_best_action(index):
    values = {i: 0 for i in range(3)}
    if index in all_qvalues:
        for action in all_qvalues[index]:
            values[action['action']] = action['qvalue']  # overrides default
        return max(values.items(), key=operator.itemgetter(1))[0]
    else:
        all_qvalues[index] = []
        return randint(0,2)


def get_index(position, speed):
    return position, speed


def main():
    global exploration_rate, measurements
    for i in range(episodes):
        observation = env.reset()
        reward = 0
        done = False
        timesteps = 0
        while not done:
            if slow and i == episodes-1: env.render()
            action = take_action(observation, reward)
            observation, reward, done, info = env.step(action)
            timesteps += 1
        print ("Episode {} finished after {} timesteps.".format(i, timesteps))
        measurements.append(timesteps)

    avg_timesteps = sum(measurements)/len(measurements)
    print("All {} episodes took {} on average".format(episodes, avg_timesteps))

    if not slow:
        with open("results.txt", "a+") as fout:
            fout.write("{} Exploration Rate {} AVG {} {}\n".format(VERSION, exploration_rate, avg_timesteps, measurements))


if __name__=='__main__':
    main()