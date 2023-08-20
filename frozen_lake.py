import gym
import numpy as np
import matplotlib.pyplot as plt

def training_plot(rewards, increment=100):
    # Get average reward of each cycle
    avg_rewards = []

    for i in range(0, len(rewards), increment):
        avg_rewards.append(np.average(rewards[i:i+increment])) 

    # Plot
    plt.plot(avg_rewards)
    plt.ylabel('Average Reward')
    plt.xlabel(f'Episodes ({increment}\'s)')
    plt.show()

def main(episodes=1500, max_steps=100, render=False, plot=False):
    # Create gym environment
    env = gym.make('FrozenLake-v1', render_mode='human')

    states = env.observation_space.n
    actions = env.action_space.n

    # Generate q-table representing env
    q_table = np.zeros((states, actions))

    # Learning rate and gamma
    lr=0.81
    gamma=0.9

    # Randomly pick valid action based epsilon, else use q-table for best action
    epsilon = 0.9

    # Training, to update q-table, Q[state,action]=Q[state,action]+α∗(reward+γ∗max(Q[newState,:])−Q[state,action])
    rewards = []
    for _ in range(episodes):

        # Reset environment after reward
        state = env.reset()[0]

        # One cycle of frozen lake
        for _ in range(max_steps):
        
            if render:
                env.render()

            # Randomly pick valid action, this probability will decrease for every reward after a training cycle
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  
            else:
                action = np.argmax(q_table[state, :])

            # Update q-table
            next_state, reward, done, _, _ = env.step(action)
            q_table[state, action] = q_table[state, action] + lr * (reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action])
            state = next_state

            # Reached goal
            if done: 
                rewards.append(reward)
                epsilon -= 0.001
                break

    print(q_table)
    print(f'Average reward: {sum(rewards)/len(rewards)}:')

    if plot:
        training_plot(rewards)

if __name__ == '__main__':
    main(render=True, plot=False)