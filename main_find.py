import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import DQN,PPO
import datetime
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from env import MySim_D

if __name__=="__main__":
    
    # DQN,A2C,HER,PPO,QR-DQN,TRPO,Maskable PPO can use
    env=Monitor(MySim_D())
    model =DQN(policy="MlpPolicy", env=env,)
    model.learn(total_timesteps=200000)
    # timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    # model.save(f'test_{timestamp}')
    model.save(f'test_DQN')
    episode_lengths = env.get_episode_lengths()
    episode_rewards = env.get_episode_rewards()
   
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(episode_lengths, label='Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Length')
    plt.title('Episode Lengths Over Time')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(episode_rewards, label='Episode Rewards', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.title('Episode Rewards Over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()


    for i in range(10):
      obs, info = env.unwrapped.reset()
      terminated = False
      truncated = False
      print(f"Starting Episode {i+1}")
      while not (terminated or truncated):
          action, _ = model.predict(observation=obs)
          obs, reward, terminated, truncated, info = env.unwrapped.step(action)

      path_xdata = env.unwrapped.path_xdata
      path_ydata = env.unwrapped.path_ydata
      # Create a figure and axis for plotting
      fig, ax = env.unwrapped.render(mode='human')
      # Line object to update in the animation
      line, = ax.plot([], [], marker='x', color='blue', label='Path')
      for obs in env.unwrapped.dynamic_obstacles:
        dynamic_obs_patches = [plt.Rectangle((ob[0][0], ob[0][1]), 1, 1, color="purple", fill=True) for ob in obs]
      for patch in dynamic_obs_patches:
        ax.add_patch(patch)

      # Update function for animation
      def update(frame):
          # Update dynamic obstacles (move obstacles)
          dynamic_obs_positions = np.array([ob[0] for ob in env.unwrapped.dynamic_obstacles[frame]])
          for i, patch in enumerate(dynamic_obs_patches):
            patch.set_xy((dynamic_obs_positions[i][1], dynamic_obs_positions[i][0]))
          line.set_data(path_ydata[:frame+1], path_xdata[:frame+1])
          return line, *dynamic_obs_patches

      # Create the animation
      ani = animation.FuncAnimation(fig, update, frames=len(path_xdata), interval=500, blit=False)

      # Display the animation
      plt.show()

      print(f"Path for Episode {i+1}: X - {path_xdata}, Y - {path_ydata}")
