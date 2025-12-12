import time
import numpy as np
import torch
import argparse
from dqn_agent import DQNAgent
from vector_env import VectorEnv

def to_tensor(states, device):
    return torch.tensor(np.array(states, dtype=np.float32), device=device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--max-steps', type=int, default=2000)  # per episode cap
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--target-update', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-path', type=str, default='dqn_flappy.pth')
    args = parser.parse_args()

    device = torch.device(args.device)
    print("Device:", device)

    envs = VectorEnv(num_envs=args.num_envs)
    agent = DQNAgent(device=device, lr=args.lr, gamma=args.gamma,
                     batch_size=args.batch_size, target_update=args.target_update)

    # epsilon schedule
    eps_start = 1.0
    eps_end = 0.05
    eps_decay_steps = 1_000_000  # slow decay over 1M global steps
    min_epsilon = 0.1           # optional minimum exploration
    global_steps = 0             # initialize BEFORE use

    states = envs.reset()  # list of state tuples
    episode_rewards = [0.0] * args.num_envs
    episode_counts = 0
    last_save = time.time()

    for ep in range(args.episodes):
        for step_in_ep in range(args.max_steps):
            # epsilon calculation
            epsilon = eps_end + (eps_start - eps_end) * max(0, 1 - global_steps / eps_decay_steps)
            epsilon = max(epsilon, min_epsilon)

            state_tensor = to_tensor(states, device)  # [N, state_dim]
            actions = agent.select_action(state_tensor, epsilon).cpu().numpy().tolist()

            next_states, rewards, dones = envs.step(actions)

            # store transitions for each env
            for i in range(args.num_envs):
                agent.store_transition(states[i], actions[i], rewards[i], next_states[i], dones[i])
                episode_rewards[i] += rewards[i]

                if dones[i]:
                    episode_counts += 1

            states = next_states
            global_steps += args.num_envs  # keep counting per env

            # optimize
            loss = agent.optimize()

            if episode_counts >= 50:
                avg_reward = sum(episode_rewards) / (len(episode_rewards) + 1e-9)
                print(f"[Ep {ep} Step {step_in_ep}] global_steps {global_steps} eps {epsilon:.3f} avg_episode_reward {avg_reward:.3f} replay_len {len(agent.replay)} loss {loss}")
                episode_counts = 0
                episode_rewards = [0.0] * args.num_envs

            # save checkpoint every 2 minutes
            if time.time() - last_save > 60 * 2:
                agent.save(args.save_path)
                last_save = time.time()

        # end of episode save
        agent.save(args.save_path)
        print(f"Saved model to {args.save_path} at epoch {ep}")

    # final save
    agent.save(args.save_path)
    print("Training finished. Model saved.")

if __name__ == "__main__":
    main()
