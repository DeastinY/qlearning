import gym
slow = False
env = gym.make("MountainCar-v0")
measurements = []
episodes = 10


def take_action(timesteps, observation, reward, done):
    act_back, act_none, act_break = range(3)
    return env.action_space.sample()

for _ in range(episodes):
    observation = env.reset()
    reward = 0
    done = False
    timesteps = 0
    while not done:
        if slow: env.render()
        action = take_action(timesteps, observation, reward, done)
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
        fout.write("Random {} AVG {}\n".format(measurements, avg_timesteps))