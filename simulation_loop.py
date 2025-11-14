from environment import HVACTrainingEnv

def simulation_loop():

    env = HVACTrainingEnv(render_mode="human", max_steps=10)

    obs, info = env.reset()
    done = False
    truncated = False

    while not (done or truncated):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()

    env.close()