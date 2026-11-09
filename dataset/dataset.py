import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

(train_data, test_data), info = tfds.load(
    "emnist/letters",
    split=["train", "test"],
    as_supervised=True,
    batch_size=-1,
    with_info=True
)


X_train, y_train = tfds.as_numpy(train_data)
X_test, y_test = tfds.as_numpy(test_data)


print("Training images:", X_train.shape)
print("Training labels:", y_train.shape)

print("Testing images:", X_test.shape)
print("Testing labels:", y_test.shape)

print("Image data type:", X_train.dtype)
print("Number of unique labels:", len(np.unique(y_train)))

print("Dataset downloaded successfully!")



# 4. Display 10 sample images
plt.figure(figsize=(12, 5))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    # Correct EMNIST image orientation
    image = np.transpose(X_train[i].squeeze())

    plt.imshow(image, cmap="gray")

    # Convert label into alphabet
    alphabet = chr(int(y_train[i]) + 64)

    plt.title(f"Label: {alphabet}")
    plt.axis("off")

plt.tight_layout()
plt.show()