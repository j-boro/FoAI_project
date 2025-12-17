import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataset_generation import generate_data

def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth))

def main():
    print("Loading model...")
    model = tf.keras.models.load_model(
        "path_planner_model_250k.keras", 
        custom_objects={"dice_loss": dice_loss}
    )

    NUM_SAMPLES = 3
    
    print(f"Generating {NUM_SAMPLES} test samples...")
    test_inputs, test_targets = generate_data(
        NUM_SAMPLES, 
        grid_size=32, 
        obstacle_density=0.4, 
        min_dist=15
    )
    
    print("Predicting paths...")
    predictions = model.predict(test_inputs)
    
    fig, axes = plt.subplots(NUM_SAMPLES, 3, figsize=(15, 5 * NUM_SAMPLES))
    
    if NUM_SAMPLES == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(NUM_SAMPLES):
        axes[i, 0].imshow(test_inputs[i, :, :, 0], cmap='gray_r')
        axes[i, 0].set_title(f"Sample {i+1}: Input")
        
        start_y, start_x = np.argwhere(test_inputs[i, :, :, 1] == 1)[0]
        end_y, end_x = np.argwhere(test_inputs[i, :, :, 2] == 1)[0]
        axes[i, 0].plot(start_x, start_y, 'go', markersize=10, label='Start')
        axes[i, 0].plot(end_x, end_y, 'ro', markersize=10, label='End')
        if i == 0: axes[i, 0].legend()

        axes[i, 1].imshow(test_targets[i, :, :, 0], cmap='gray')
        axes[i, 1].set_title("Ground Truth")

        axes[i, 2].imshow(predictions[i, :, :, 0], cmap='magma')
        axes[i, 2].set_title("Prediction")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()