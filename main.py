import operator
import asyncio
from random import random, randint
from multiprocessing.pool import ThreadPool
import gym
from bokeh.charts import output_file
from bokeh.plotting import figure

plot = False
render_last = True
env = gym.make("MountainCar-v0")
VERSION = "V1.6"
measurements = []
exploration_rate = 0.3
exploration_decay = 0.999
episodes = 100
bucket_brigade_discount = 0.999
stop_at = 150
all_qvalues = {}  # { '(position, speed)': [{'action': {'Action' : A, 'Qvalue': Q, 'ResultState': (position, speed)]}}
state_relations = {}  # {'Resultstate' : ['(position, speed)', ... ]
path = []
last_action, last_position, last_precise_position, last_speed, last_index = [None for _ in range(5)]


def take_action(observation, reward):
    # 0 = back, 1 = nothing, 2 = front
    global last_action, last_position, last_speed, last_index, \
        last_precise_position, exploration_decay, exploration_rate
    precise_position = observation[0]
    position = discretize_position(precise_position)
    speed = 0 if last_action is None else discretize_speed(calculate_speed(last_precise_position, precise_position))
    index = get_index(position, speed)
    exploration_rate *= exploration_decay

    if last_action is None:
        action = randint(0, 2)
        last_index, last_action = index, action
        all_qvalues[index] = [{'action': action, 'qvalue': reward, 'resultstate': None}]
    else:
        last_index = get_index(last_position, last_speed)
        if random() < exploration_rate:
            action = randint(0, 2)
            all_qvalues[index] = []
        else:
            action = get_best_action(index)

    update_state(last_index, last_action, reward + best_qvalue(index), index)
    last_action, last_position, last_precise_position, last_index, last_speed = \
        action, position, precise_position, index, speed
    path.append((last_index, last_action))
    return action


def reverse_update():
    # Do a complete reverse update of the path. Update the values of the path and every state that leads to one
    # of the paths states. (Note to self : This can't be parallelized well, stupid !)
    pool = ThreadPool(8)
    loop = asyncio.get_event_loop()
    for p in path:
        index, action = p
        part_reverse_update(index, action, loop, pool, True)


@asyncio.coroutine
def start_threads(pool, threading_data):
    pool.map(part_reverse_update_star, threading_data)


def part_reverse_update_star(raw_data):
    part_reverse_update(*raw_data)


def part_reverse_update(index, action, loop, pool, relation_update):
    for a in all_qvalues[index]:
        if a['action'] == action:
            resultstate = a['resultstate']
            old_q = a['qvalue']
            new_q = -1 + best_qvalue(resultstate)
            qvalue = (old_q + bucket_brigade_discount * new_q) / 2
            update_state(index, action, qvalue, resultstate)
    if relation_update:
        previous_states = state_relations[index]
        threading_data = [idx_act+(loop, pool, False) for idx_act in previous_states]
        loop.run_until_complete(start_threads, )


def update_state(index, action, qvalue, resultstate):
    if resultstate not in state_relations:
        state_relations[resultstate] = []
    state_relations[resultstate].append((index, action))
    for a in all_qvalues[index]:
        if a['action'] == action:
            a['resultstate'], a['qvalue'] = resultstate, qvalue
            break
    else:
        all_qvalues[index].append({'action': action, 'qvalue': qvalue, 'resultstate': resultstate})


def best_qvalue(index):
    max_q = None
    if index in all_qvalues:
        for a in all_qvalues[index]:
            q = a['qvalue']
            max_q = q if max_q is None or q > max_q else max_q
    return max_q if max_q is not None else 0


def calculate_speed(old_position, position):
    return old_position - position


def discretize_position(position):
    # -1.2 to 0.6
    return int(position * 100)


def discretize_speed(speed):
    # -0.07 to 0.07
    return int(speed * 1000)


def get_best_action(index):
    values = {i: 0 for i in range(3)}
    if index in all_qvalues:
        for action in all_qvalues[index]:
            values[action['action']] = action['qvalue']  # overrides default
        return max(values.items(), key=operator.itemgetter(1))[0]
    else:
        all_qvalues[index] = []
        return randint(0, 2)


def get_index(position, speed):
    return position, speed


def plot_results():
    output_file('plot.html')
    position = [i[0] for i in all_qvalues]
    speed = [i[1] for i in all_qvalues]
    qvalues = [[j['qvalue'] for j in i] for i in all_qvalues.values()]
    qvalues = [max(i) if len(i) > 0 else print(i) for i in qvalues]
    df = {'Position': position, 'Speed': speed, 'QValues': qvalues}
    # hm = HeatMap(df, x='Position', y='Speed', values='QValues', stat=None)
    f = figure()
    for p in position:
        for s in speed:
            idx = get_index(p, s)
            if idx not in all_qvalues:
                continue
            q = all_qvalues[idx]
            q = max([i['qvalue'] for i in q])
            if q > -5:
                f.circle(p, s)


def main():
    global exploration_rate, measurements, path
    last_iteration = False
    for i in range(episodes):
        observation = env.reset()
        reward, timesteps, done = 0, 0, False
        while not done:
            if last_iteration or (render_last and i == episodes - 1):
                env.render()
            action = take_action(observation, reward)
            observation, reward, done, info = env.step(action)
            timesteps += 1
        reverse_update()
        print("Episode {} finished after {} timesteps.".format(i, timesteps))
        measurements.append(timesteps)
        if last_iteration:
            break
        if timesteps < stop_at:
            last_iteration = True

    avg_timesteps = sum(measurements) / len(measurements)
    print("All {} episodes took {} on average".format(i+1, avg_timesteps))
    with open("results.txt", "a+") as fout:
        fout.write("{} Exploration Rate {} AVG {} {}\n".format(VERSION, exploration_rate, avg_timesteps, measurements))
    if plot:
        plot_results()


if __name__ == '__main__':
    main()
