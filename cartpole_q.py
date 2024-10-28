import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
from IPython import display

def run(is_training=True, render=False, plot_interval=50):
    """
    Run the CartPole environment with Q-learning.
    
    Args:
        is_training (bool): If True, train the agent. If False, load and run a trained model.
        render (bool): If True, render the environment visually.
        plot_interval (int): Update the training plots every n episodes.
    """
    # Initialize environment
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    
    # Set up state space discretization
    # These values represent the range of each state variable in the CartPole environment
    STATE_SPACES = {
        'cart_position': {'min': -2.4, 'max': 2.4},      # Position of cart on track
        'cart_velocity': {'min': -4.0, 'max': 4.0},      # Velocity of cart
        'pole_angle': {'min': -0.2095, 'max': 0.2095},   # Angle of pole (in radians)
        'pole_velocity': {'min': -4.0, 'max': 4.0}       # Angular velocity of pole
    }
    
    # Number of bins for discretizing each state variable
    BINS = 10
    
    # Create discretized spaces for each state variable
    discretized_spaces = {
        name: np.linspace(space['min'], space['max'], BINS)
        for name, space in STATE_SPACES.items()
    }

    # Initialize or load Q-table
    state_dimensions = tuple(len(space) + 1 for space in discretized_spaces.values())
    if is_training:
        # Initialize Q-table with small random values for exploration
        Q_table = np.random.uniform(
            low=-0.1, 
            high=0.1, 
            size=state_dimensions + (env.action_space.n,)
        )
    else:
        try:
            with open('cartpole_q.pkl', 'rb') as f:
                Q_table = pickle.load(f)
            print("Loaded previously trained Q-table")
        except FileNotFoundError:
            print("Error: No trained model found. Please run training first.")
            return
    
    # Training hyperparameters
    LEARNING_PARAMS = {
        'alpha': 0.2,           # Learning rate: how much to update Q-values (0-1)
        'gamma': 0.99,          # Discount factor: importance of future rewards (0-1)
        'epsilon': 1.0,         # Initial exploration rate
        'epsilon_decay': 0.999, # How much to decrease epsilon each episode
        'epsilon_min': 0.01,    # Minimum exploration rate
        'max_steps': 10000,     # Maximum steps per episode
        'target_reward': 1000   # Target reward for considering training successful
    }

    # Initialize training metrics
    rewards_history = []
    epsilon_history = []
    episodes = 0
    
    # Set up live plotting
    if is_training:
        plt.ion()  # Enable interactive plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('Training Progress')
        
        # Initialize plot lines
        rewards_line, = ax1.plot([], [], 'b-', label='Rewards')
        mean_rewards_line, = ax1.plot([], [], 'r-', label='Mean Rewards (100 ep)')
        epsilon_line, = ax2.plot([], [], 'g-', label='Epsilon')
        
        # Set up plot layouts
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Rewards per Episode')
        ax1.legend()
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Epsilon')
        ax2.set_title('Exploration Rate Decay')
        ax2.legend()
        
        plt.tight_layout()

    def discretize_state(state):
        """Convert continuous state values to discrete indices."""
        discretized = [
            np.digitize(s, space)
            for s, space in zip(state, discretized_spaces.values())
        ]
        return tuple(discretized)

    def update_plots():
        """Update the training progress plots."""
        if len(rewards_history) > 0:
            episodes_x = list(range(len(rewards_history)))
            
            # Update rewards plot
            ax1.set_xlim(0, len(rewards_history))
            ax1.set_ylim(0, max(rewards_history) + 100)
            rewards_line.set_data(episodes_x, rewards_history)
            
            # Calculate and plot moving average
            if len(rewards_history) >= 100:
                moving_avg = [np.mean(rewards_history[max(0, i-100):i+1]) 
                            for i in range(len(rewards_history))]
                mean_rewards_line.set_data(episodes_x, moving_avg)
            
            # Update epsilon plot
            ax2.set_xlim(0, len(epsilon_history))
            ax2.set_ylim(0, 1)
            epsilon_line.set_data(episodes_x, epsilon_history)
            
            plt.draw()
            plt.pause(0.01)

    # Main training/evaluation loop
    while True:
        state = env.reset()[0]
        current_state = discretize_state(state)
        episode_reward = 0
        done = False

        # Run single episode
        while not done and episode_reward < LEARNING_PARAMS['max_steps']:
            # Choose action using epsilon-greedy policy
            if is_training and np.random.random() < LEARNING_PARAMS['epsilon']:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(Q_table[current_state])  # Exploit

            # Take action and observe result
            new_state, reward, done, _, _ = env.step(action)
            new_state = discretize_state(new_state)
            episode_reward += reward

            # Update Q-table during training
            if is_training:
                best_future_value = np.max(Q_table[new_state])
                current_value = Q_table[current_state + (action,)]
                
                # Q-learning update rule
                Q_table[current_state + (action,)] += LEARNING_PARAMS['alpha'] * (
                    reward + LEARNING_PARAMS['gamma'] * best_future_value - current_value
                )

            current_state = new_state

            # Print progress during evaluation
            if not is_training and episode_reward % 100 == 0:
                print(f'Episode {episodes} - Current Reward: {episode_reward}')

        # Post-episode updates
        rewards_history.append(episode_reward)
        if is_training:
            epsilon_history.append(LEARNING_PARAMS['epsilon'])
            
            # Update live plots every plot_interval episodes
            if episodes % plot_interval == 0:
                update_plots()
            
            # Print training progress
            if episodes % 100 == 0:
                mean_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
                print(f'Episode: {episodes} | '
                      f'Reward: {episode_reward:.1f} | '
                      f'Epsilon: {LEARNING_PARAMS["epsilon"]:.2f} | '
                      f'Mean Reward: {mean_reward:.1f}')

            # Decay exploration rate
            LEARNING_PARAMS['epsilon'] = max(
                LEARNING_PARAMS['epsilon'] * LEARNING_PARAMS['epsilon_decay'],
                LEARNING_PARAMS['epsilon_min']
            )

        # Check if training is complete
        if len(rewards_history) >= 100:
            mean_reward = np.mean(rewards_history[-100:])
            if mean_reward > LEARNING_PARAMS['target_reward'] or episodes >= 10000:
                break

        episodes += 1

    env.close()

    # Save results if training
    if is_training:
        print("\nSaving training results...")
        
        # Save Q-table
        with open('cartpole_q.pkl', 'wb') as f:
            pickle.dump(Q_table, f)
        
        # Save final plot
        plt.savefig('cartpole_training_results.png')
        plt.close()
        
        print(f"Training completed after {episodes} episodes")
        print(f"Final average reward: {mean_reward:.1f}")
        print("Q-table and training plots have been saved")

if __name__ == '__main__':
    # run(is_training=True, render=False, plot_interval=50)
    run(is_training=False, render=True)