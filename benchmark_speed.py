"""
Quick benchmark: measures real frames/sec for the DQN-on-Pong training loop
on this machine. Run this directly in a normal Terminal window (NOT through
any sandboxed tool) for an accurate number.

Usage:
    python3 -m venv venv
    source venv/bin/activate
    pip install gymnasium ale-py opencv-python-headless torch
    python benchmark_speed.py
"""
import collections
import time

import numpy as np
import torch
from torch import nn, optim

from lib import dqn_model, wrappers

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
REPLAY_START_SIZE = 2000   # lower warmup so the benchmark reaches training steps quickly
SYNC_TARGET_FRAMES = 1000
LEARNING_RATE = 1e-4
N_FRAMES = 8000

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=bool), np.array(next_states))


def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch
    states_v = torch.as_tensor(states, dtype=torch.float32).to(device)
    next_states_v = torch.as_tensor(next_states, dtype=torch.float32).to(device)
    actions_v = torch.as_tensor(actions, dtype=torch.int64).to(device)
    rewards_v = torch.as_tensor(rewards, dtype=torch.float32).to(device)
    done_mask = torch.as_tensor(dones, dtype=torch.bool).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def main():
    device = torch.device("cpu")
    print("device:", device)

    env = wrappers.make_env("ALE/Pong-v5")
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    buffer = ExperienceBuffer(REPLAY_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    state, _ = env.reset()
    warmup_done_at = None
    t_start = time.time()
    t_after_warmup = None

    for frame_idx in range(1, N_FRAMES + 1):
        epsilon = max(0.02, 1.0 - frame_idx / 100000)
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_v = torch.as_tensor(np.array([state]), dtype=torch.float32).to(device)
            with torch.no_grad():
                q_vals_v = net(state_v)
            action = int(torch.argmax(q_vals_v, dim=1).item())

        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.append(Experience(state, action, reward, done, new_state))
        state = new_state
        if done:
            state, _ = env.reset()

        if len(buffer) < REPLAY_START_SIZE:
            continue
        if warmup_done_at is None:
            warmup_done_at = frame_idx
            t_after_warmup = time.time()

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()

    t_end = time.time()
    total_elapsed = t_end - t_start
    print(f"Total: {N_FRAMES} frames in {total_elapsed:.2f}s -> {N_FRAMES / total_elapsed:.2f} f/s "
          f"(includes replay-warmup, env-stepping-only phase)")

    if warmup_done_at is not None:
        steady_frames = N_FRAMES - warmup_done_at
        steady_elapsed = t_end - t_after_warmup
        steady_fps = steady_frames / steady_elapsed
        print(f"Steady-state (post warmup, includes training steps): "
              f"{steady_frames} frames in {steady_elapsed:.2f}s -> {steady_fps:.2f} f/s")
        # book's baseline typically needs roughly 400k-1M frames to solve Pong
        for target in (400_000, 700_000, 1_000_000):
            est_minutes = target / steady_fps / 60
            print(f"  -> at this speed, {target:,} frames would take ~{est_minutes:.1f} minutes")


if __name__ == "__main__":
    main()
