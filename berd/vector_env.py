from env import FlappyEnv

class VectorEnv:
    def __init__(self, num_envs=16):
        self.num_envs = num_envs
        self.envs = [FlappyEnv() for _ in range(num_envs)]

    def reset(self):
        return [env.reset() for env in self.envs]

    def step(self, actions):
        """
        actions: list of ints with length = num_envs
        """
        states = []
        rewards = []
        dones = []

        for env, action in zip(self.envs, actions):
            s, r, d = env.step(action)
            # Auto-reset dead envs so training continues
            if d:
                s = env.reset()
            states.append(s)
            rewards.append(r)
            dones.append(d)

        return states, rewards, dones
