import numpy as np
import matplotlib.pyplot as plt
import random

# Parameters
grid_size = 10
episodes = 500
alpha = 0.1
gamma = 0.9
epsilon = 0.2
max_steps = 100

EMPTY = 0
START = 1
GOAL = 2
NO_FLY = -1

# Grid and wind setup
def create_grid():
    grid = np.zeros((grid_size, grid_size), dtype=int)
    grid[0, 0] = START
    grid[grid_size - 1, grid_size - 1] = GOAL
    for _ in range(15):
        x, y = np.random.randint(0, grid_size, 2)
        if grid[x, y] == EMPTY:
            grid[x, y] = NO_FLY
    return grid

def create_wind_map(grid):
    wind_map = {}
    directions = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
    }
    for _ in range(10):
        x, y = np.random.randint(0, grid_size, 2)
        if grid[x, y] == EMPTY:
            direction = random.choice(list(directions.keys()))
            wind_map[(x, y)] = directions[direction]
    return wind_map

def get_start_pos(grid):
    return tuple(np.argwhere(grid == START)[0])

def get_goal_pos(grid):
    return tuple(np.argwhere(grid == GOAL)[0])

def is_valid(pos):
    x, y = pos
    return 0 <= x < grid_size and 0 <= y < grid_size

def get_next_state(state, action):
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    next_state = (state[0] + moves[action][0], state[1] + moves[action][1])
    return next_state if is_valid(next_state) else state

def apply_wind(state, wind_map):
    if state in wind_map:
        dx, dy = wind_map[state]
        x, y = state[0] + dx, state[1] + dy
        if 0 <= x < grid_size and 0 <= y < grid_size:
            return (x, y)
    return state

def get_reward(state, grid, use_human_shaping=False):
    if grid[state] == GOAL:
        return 10
    elif grid[state] == NO_FLY:
        return -10
    else:
        if use_human_shaping:
            x, y = state
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        if grid[nx, ny] == NO_FLY:
                            return -3
        return -1

def init_q_table():
    return np.zeros((grid_size, grid_size, 4))

def train_agent(grid, use_human_shaping=False, use_wind=False):
    q_table = init_q_table()
    rewards_per_episode = []
    crash_log = []
    steps_to_goal = []
    wind_interference_count = 0
    wind_map = create_wind_map(grid) if use_wind else {}

    for ep in range(episodes):
        state = get_start_pos(grid)
        total_reward = 0
        crashed = False
        steps = 0

        if use_wind and ep % 50 == 0:
            wind_map = create_wind_map(grid)

        for step in range(max_steps):
            action = np.random.choice(4) if random.uniform(0, 1) < epsilon else np.argmax(q_table[state[0], state[1]])
            next_state = get_next_state(state, action)
            wind_pushed_state = apply_wind(next_state, wind_map)
            if wind_pushed_state != next_state:
                wind_interference_count += 1
            next_state = wind_pushed_state

            reward = get_reward(next_state, grid, use_human_shaping)
            if grid[next_state] == NO_FLY:
                crashed = True

            total_reward += reward

            old_value = q_table[state[0], state[1], action]
            next_max = np.max(q_table[next_state[0], next_state[1]])
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[state[0], state[1], action] = new_value

            state = next_state
            steps += 1
            if grid[state] == GOAL:
                steps_to_goal.append(steps)
                break

        rewards_per_episode.append(total_reward)
        crash_log.append(int(crashed))

    avg_reward = np.mean(rewards_per_episode)
    avg_steps = np.mean(steps_to_goal) if steps_to_goal else float('inf')
    total_crashes = sum(crash_log)
    path_optimality = (abs(grid_size - 1) * 2) / avg_steps if avg_steps != 0 else 0
    wrs = 1 - (wind_interference_count / (episodes * max_steps))
    avg_deviation = wind_interference_count / episodes

    print("\n--- Performance Summary ---")
    print(f"Avg. Total Reward: {avg_reward:.2f}")
    print(f"Avg. Steps to Goal: {avg_steps:.2f}")
    print(f"Total Crashes: {total_crashes}")
    print(f"Path Optimality (Final): {path_optimality:.2f}")
    print(f"Wind Resistance Score (WRS): {wrs:.2f}")
    print(f"Avg. Deviation per Ep.: {avg_deviation:.2f}")

    return q_table, rewards_per_episode, crash_log, wind_interference_count

def visualize_path(q_table, grid, title):
    state = get_start_pos(grid)
    path = [state]
    for _ in range(50):
        action = np.argmax(q_table[state[0], state[1]])
        next_state = get_next_state(state, action)
        if next_state == state or grid[next_state] == NO_FLY:
            break
        path.append(next_state)
        state = next_state
        if grid[state] == GOAL:
            break
    display = np.copy(grid).astype(str)
    display[display == '0'] = '.'
    display[display == '-1'] = 'X'
    display[display == '1'] = 'S'
    display[display == '2'] = 'G'
    for x, y in path[1:-1]:
        display[x, y] = '*'
    print(f"\n{title} Path Visualization:\n")
    for row in display:
        print(" ".join(row))

def plot_rewards_and_crashes(rewards_q, rewards_human, crashes_q, crashes_human):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(rewards_q, label="Q-learning")
    ax[0].plot(rewards_human, label="Human-guided")
    ax[0].set_title("Total Reward per Episode")
    ax[0].set_ylabel("Reward")
    ax[0].legend()
    ax[0].grid(True)
    ax[1].plot(np.cumsum(crashes_q), label="Q-learning")
    ax[1].plot(np.cumsum(crashes_human), label="Human-guided")
    ax[1].set_title("Cumulative Crashes into No-Fly Zones")
    ax[1].set_ylabel("Crashes")
    ax[1].set_xlabel("Episode")
    ax[1].legend()
    ax[1].grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    grid = create_grid()

    print("Training without wind...")
    q1, rewards_q, crashes_q, _ = train_agent(grid, use_human_shaping=False, use_wind=False)

    print("Training with wind and human guidance...")
    q2, rewards_human, crashes_human, wind_events = train_agent(grid, use_human_shaping=True, use_wind=True)

    visualize_path(q1, grid, "Q-Learning")
    visualize_path(q2, grid, "Human-Guided Q-Learning with Wind")
    plot_rewards_and_crashes(rewards_q, rewards_human, crashes_q, crashes_human)

    episodes_with_wind = episodes
    WRS = 1 - (wind_events / (episodes_with_wind * max_steps))
    print(f"\nWind Resistance Score (WRS): {WRS:.2f}")
