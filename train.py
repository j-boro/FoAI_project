import numpy as np
import tensorflow as tf
import wandb
from wandb.integration.keras import WandbMetricsLogger

from dataset_generation import generate_data
from model import build_unet

CONFIG = {
    "grid_size": 32,
    "batch_size": 128,
    "epochs": 250,
    "train_samples": 250000,
    "val_samples": 25000,
    "learning_rate": 0.001,
    "obstacle_density": 0.4,
    "min_distance": 15
}

def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth))

class PathVizCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_inputs, val_targets):
        super().__init__()
        self.val_inputs = val_inputs
        self.val_targets = val_targets

    def on_epoch_end(self, epoch, logs=None):
        indices = np.random.choice(len(self.val_inputs), 3, replace=False)
        subset_inputs = self.val_inputs[indices]
        subset_targets = self.val_targets[indices]
        preds = self.model.predict(subset_inputs, verbose=0)
        
        viz_images = []
        for i in range(len(subset_inputs)):
            obstacles = subset_inputs[i, :, :, 0]
            truth = subset_targets[i, :, :, 0]
            pred_mask = preds[i, :, :, 0]
            
            viz_images.append(wandb.Image(
                obstacles,
                masks={
                    "predictions": {"mask_data": pred_mask > 0.5, "class_labels": {1: "Pred Path"}},
                    "ground_truth": {"mask_data": truth, "class_labels": {1: "True Path"}}
                },
                caption=f"Epoch {epoch}"
            ))
        wandb.log({"Validation/Visualizations": viz_images})

def main():
    wandb.init(project="simple-path-planner", config=CONFIG)
    
    print(f"Generating Data (Density: {CONFIG['obstacle_density']}, Min Dist: {CONFIG['min_distance']})...")
    X_train, y_train = generate_data(
        CONFIG["train_samples"], 
        CONFIG["grid_size"],
        obstacle_density=CONFIG["obstacle_density"],
        min_dist=CONFIG["min_distance"]
    )
    
    X_val, y_val = generate_data(
        CONFIG["val_samples"], 
        CONFIG["grid_size"],
        obstacle_density=CONFIG["obstacle_density"],
        min_dist=CONFIG["min_distance"]
    )
    
    model = build_unet(CONFIG["grid_size"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG["learning_rate"]),
        loss=dice_loss, 
        metrics=['accuracy']
    )

    callbacks = [
        WandbMetricsLogger(),
        PathVizCallback(X_val, y_val),
        
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
        ),
        
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, verbose=1
        )
    ]

    print("Starting Training...")
    model.fit(
        X_train, y_train,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    
    model.save("path_planner_model.keras")
    wandb.finish()

if __name__ == "__main__":
    main()