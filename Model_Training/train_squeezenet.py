import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
import tf_keras

def build_model(base_model_path, num_classes=26, fine_tune_layers=3):
    """
    Loads base SqueezeNet model and replaces the final classification layer
    for num_classes output classes.
    fine_tune_layers: unfreezes the last N fire modules for higher capacity.
    """
    print(f"[Model] Loading base SqueezeNet model from {base_model_path}...")
    base_model = tf_keras.models.load_model(base_model_path)
    
    # Locate the dropout layer before the classification head
    dropout_output = base_model.get_layer("dropout").output
    
    # Replace last conv and pooling for the new number of classes
    x = tf_keras.layers.Conv2D(num_classes, (1, 1), padding="valid", name="new_last_conv")(dropout_output)
    x = tf_keras.layers.Activation("relu", name="new_activation_relu")(x)
    x = tf_keras.layers.GlobalAveragePooling2D(name="new_global_avg_pool")(x)
    output = tf_keras.layers.Activation("softmax", name="new_softmax_output")(x)
    
    new_model = tf_keras.models.Model(inputs=base_model.input, outputs=output)
    
    # Trainable configuration: classification head + optionally top fire modules
    trainable_subnames = ["new_last_conv", "new_activation_relu", "new_global_avg_pool", "new_softmax_output"]
    if fine_tune_layers > 0:
        # Fire modules 8 and 9 (conv2d_20 through conv2d_24)
        trainable_subnames += ["conv2d_21", "conv2d_22", "conv2d_23", "conv2d_24"]

    for layer in new_model.layers:
        layer.trainable = any(s in layer.name for s in trainable_subnames)
                
    return new_model

def get_dataset(data_dir, img_size=(224, 224), batch_size=32, validation_split=0.2):
    """
    Loads image data directly from directory structure where each subfolder is a class.
    Caches data in memory/disk for accelerated subsequent epochs.
    """
    print(f"[Dataset] Loading data from {data_dir}...")
    train_ds = tf_keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=1337,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical"
    )

    val_ds = tf_keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=1337,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical"
    )

    class_names = train_ds.class_names
    print(f"[Dataset] Detected {len(class_names)} classes: {class_names}")

    # Standardize input scaling to [0, 1] as expected by the pipeline
    rescaling = tf_keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (rescaling(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (rescaling(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    # In-memory caching & prefetching for performance
    train_ds = train_ds.cache().shuffle(buffer_size=1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names

def train(epochs=15, batch_size=32, learning_rate=0.001):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "dataset", "raw")
    base_model_path = os.path.join(base_dir, "App", "final_model")
    save_model_dir = os.path.join(base_dir, "App", "final_model_26")
    save_json_path = os.path.join(base_dir, "App", "model_class_26.json")

    train_ds, val_ds, class_names = get_dataset(data_dir, batch_size=batch_size)
    num_classes = len(class_names)

    model = build_model(base_model_path, num_classes=num_classes)
    model.compile(
        optimizer=tf_keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    callbacks = [
        tf_keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        tf_keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)
    ]

    print(f"[Training] Starting training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    val_acc = max(history.history["val_accuracy"])
    print(f"[Training] Training completed. Best validation accuracy: {val_acc:.4f}")

    print(f"[Saving] Saving trained model to {save_model_dir}...")
    model.save(save_model_dir)

    # Create mapping: index -> uppercase letter
    class_map = {str(i): name.upper() for i, name in enumerate(class_names)}
    with open(save_json_path, "w") as f:
        json.dump(class_map, f, indent=4)
    print(f"[Saving] Saved class mapping ({num_classes} classes) to {save_json_path}")

    return model, class_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ISL SqueezeNet model on 26 letters")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
