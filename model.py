from tensorflow.keras import layers, models

def build_unet(grid_size=32):
    inputs = layers.Input(shape=(grid_size, grid_size, 3))

    # --- Encoder ---
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    p1 = layers.Dropout(0.2)(p1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    p2 = layers.Dropout(0.2)(p2)
    
    # --- Bottleneck ---
    b = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    b = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(b)

    # --- Decoder ---
    u1 = layers.UpSampling2D((2, 2))(b)
    concat1 = layers.Concatenate()([u1, c2]) 
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat1)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)

    u2 = layers.UpSampling2D((2, 2))(c3)
    concat2 = layers.Concatenate()([u2, c1])
    c4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(concat2)
    c4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c4)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c4)

    return models.Model(inputs, outputs, name="PathPlanner_UNet")