import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

def generate_heatmap(episode, grid_size, model_path):
    # Load the model
    model = load_model(model_path)

    goal_location = (22, 6)  # Center of the grid

    # Initialize an array to store the color intensities
    heatmap_data = np.zeros((grid_size, grid_size, 3))

    # Define colors for each action
    colors = {
        0: np.array([0, 0, 1]),  # Blue for up
        1: np.array([1, 0, 0]),  # Red for down
        2: np.array([0, 1, 0]),  # Green for left
        3: np.array([1, 1, 0])   # Yellow for right
    }

    # Calculate Q-values for each state and determine the color intensity
    for x in range(grid_size):
        for y in range(grid_size):
            relative_distance = (x - goal_location[0], y - goal_location[1])
            state = np.array([*relative_distance]).reshape(1, -1)
            q_values = model.predict(state)
            best_action = np.argmax(q_values)
            if (x, y) == goal_location:
                heatmap_data[x, y] = np.array([1, 1, 1])
            else:
                heatmap_data[x, y] = colors[best_action]

    # Plotting the heatmap
    plt.imshow(heatmap_data, interpolation='nearest')
    plt.xlabel(f'Episode: {episode + 1}')
    plt.axis('off')
    plt.tight_layout(pad=0)

    plt.savefig(f'./figures/heatmap_{grid_size}_{episode}', bbox_inches='tight')
