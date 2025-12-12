import random

WIDTH = 700
HEIGHT = 600

GRAVITY = 0.5
JUMP_POWER = -8
PIPE_SPEED = 3
GAP = 200

class FlappyEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.bird_y = HEIGHT // 2
        self.bird_vel = 0
        self.pipe_x = WIDTH
        self.pipe_top = random.randint(50, HEIGHT - GAP - 50)
        self.pipe_bottom = self.pipe_top + GAP
        self.score = 0
        self.done = False

        return self._get_state()

    def step(self, action):
        """
        action = 0 → do nothing
        action = 1 → jump
        """

        if self.done:
            return self.reset(), 0, True

        # Apply action
        if action == 1:
            self.bird_vel = JUMP_POWER

        # Physics
        self.bird_vel += GRAVITY
        self.bird_y += self.bird_vel

        # Move pipe
        self.pipe_x -= PIPE_SPEED

        reward = 0.01  # tiny survival reward per frame

        # Reward for being at pipe gap level (0..1)
        gap_center = (self.pipe_top + self.pipe_bottom) / 2
        distance_to_gap = abs(self.bird_y - gap_center)
        reward += max(0, 1 - (distance_to_gap / GAP))  # closer to center = higher reward

        # Reward for passing a pipe
        if self.pipe_x < -50:  # pipe passed
            self.pipe_x = WIDTH
            self.pipe_top = random.randint(50, HEIGHT - GAP - 50)
            self.pipe_bottom = self.pipe_top + GAP
            self.score += 1
            reward += 5.0  # pipe bonus

        # Collision check
        if (
            self.bird_y <= 0 or
            self.bird_y >= HEIGHT or
            (self.bird_y < self.pipe_top and 100 > self.pipe_x and 100 < self.pipe_x + 40) or
            (self.bird_y > self.pipe_bottom and 100 > self.pipe_x and 100 < self.pipe_x + 40)
        ):
            self.done = True
            reward = -5  # penalty for dying

        return self._get_state(), reward, self.done


    def _get_state(self):
        return (
            self.bird_y,
            self.bird_vel,
            self.pipe_x,
            self.pipe_top,
            self.pipe_bottom,
        )
