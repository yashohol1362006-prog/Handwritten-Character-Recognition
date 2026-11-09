
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import classification_report

# Configuration
EPOCHS = 10
BATCH_SIZE = 128

os.makedirs("models", exist_ok=True)

print("Loading EMNIST dataset...")

(train_data, test_data) = tfds.load(
    "emnist/letters",
    split=["train", "test"],
    as_supervised=True,
    batch_size=-1
)

X_train, y_train = tfds.as_numpy(train_data)
X_test, y_test = tfds.as_numpy(test_data)

# Convert images into correct orientation
X_train = np.transpose(X_train, (0, 2, 1, 3))
X_test = np.transpose(X_test, (0, 2, 1, 3))

# Normalize images
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Convert labels from 1-26 to 0-25
y_train = y_train - 1
y_test = y_test - 1

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# CNN Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(26, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")
print("Test Loss:", loss)

# Classification report
predictions = model.predict(X_test, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

print("\nClassification Report:")
print(classification_report(
    y_test,
    predicted_labels,
    target_names=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
))

# Save model
model.save("models/character_model.keras")

# Save training history
np.save("models/training_history.npy", history.history)

print("\nModel saved successfully!")
