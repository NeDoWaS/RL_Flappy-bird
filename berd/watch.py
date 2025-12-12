import pygame
import sys
import torch
import numpy as np
from dqn_agent import QNetwork
import env as flappy_env  # to access WIDTH/HEIGHT constants

pygame.init()

WIDTH, HEIGHT = flappy_env.WIDTH, flappy_env.HEIGHT
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
FONT = pygame.font.SysFont(None, 32)

# Game constants (same as RL env)
GRAVITY = flappy_env.GRAVITY
JUMP_POWER = flappy_env.JUMP_POWER
PIPE_SPEED = flappy_env.PIPE_SPEED
GAP = flappy_env.GAP

class Bird:
    def __init__(self):
        self.x = 80
        self.y = HEIGHT // 2
        self.vel = 0
        self.width = 20
        self.height = 20

    def update(self):
        self.vel += GRAVITY
        self.y += self.vel

    def jump(self):
        self.vel = JUMP_POWER

    def rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

class Pipe:
    def __init__(self):
        self.x = WIDTH
        self.top = np.random.randint(50, HEIGHT - GAP - 50)
        self.bottom = self.top + GAP
        self.width = 40

    def update(self):
        self.x -= PIPE_SPEED

    def rects(self):
        return (
            pygame.Rect(self.x, 0, self.width, self.top),
            pygame.Rect(self.x, self.bottom, self.width, HEIGHT - self.bottom)
        )

def get_state(bird, pipe):
    return np.array([
        bird.y,
        bird.vel,
        pipe.x,
        pipe.top,
        pipe.bottom
    ], dtype=np.float32)

def load_model(path, device):
    model = QNetwork()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("dqn_flappy.pth", device)
    print("Loaded model:", device)

    while True:
        bird = Bird()
        pipes = [Pipe()]
        score = 0

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Auto pipe spawn
            if pipes[-1].x < WIDTH - 200:
                pipes.append(Pipe())

            # Move bird and pipes
            bird.update()
            for pipe in pipes:
                pipe.update()

            # Remove old pipes
            if pipes[0].x < -50:
                pipes.pop(0)
                score += 1

            # Choose closest pipe
            pipe = pipes[0]

            # RL state
            state = get_state(bird, pipe)
            state_tensor = torch.tensor(state, device=device).unsqueeze(0)

            with torch.no_grad():
                qvals = model(state_tensor)
                action = int(torch.argmax(qvals).item())

            if action == 1:
                bird.jump()

            # Collision detection
            dead = False
            for rect in pipe.rects():
                if bird.rect().colliderect(rect):
                    dead = True

            if bird.y <= 0 or bird.y >= HEIGHT:
                dead = True

            # Drawing
            screen.fill((150, 200, 255))
            pygame.draw.rect(screen, (255, 255, 0), bird.rect())

            for p in pipes:
                for rect in p.rects():
                    pygame.draw.rect(screen, (0, 200, 0), rect)

            text = FONT.render(f"Score: {score}", True, (0, 0, 0))
            screen.blit(text, (10, 10))

            pygame.display.update()

            if dead:
                break

            clock.tick(60)

if __name__ == "__main__":
    main()
