import matplotlib.pyplot as plt
from dataset_generation import generate_data

CONFIG = {
    "grid_size": 32,
    "obstacle_density": 0.4,
    "min_distance": 15
}

print("Generating small batch...")
inputs, targets = generate_data(5, CONFIG["grid_size"], CONFIG["obstacle_density"], CONFIG["min_distance"])

print("Displaying samples (Close window to exit)...")
fig, ax = plt.subplots(5, 2, figsize=(5, 15))
for i in range(5):
    ax[i,0].imshow(inputs[i,:,:,0], cmap='gray_r')
    ax[i,0].set_title(f"Sample {i} Map")
    ax[i,1].imshow(targets[i,:,:,0], cmap='gray')
    ax[i,1].set_title(f"Sample {i} Path")
plt.tight_layout()
plt.show()