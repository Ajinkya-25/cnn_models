import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

data_train = tf.keras.preprocessing.image_dataset_from_directory(
    "F:/datasets/plants",
    validation_split=0.2,
    subset="training",
    shuffle=True,
    seed=42,
    image_size=(256, 256),
    batch_size=32,
)

data_test = tf.keras.preprocessing.image_dataset_from_directory(
    "F:/datasets/plants",
    validation_split=0.2,
    subset="validation",
    shuffle=True,
    seed=42,
    image_size=(256, 256),
    batch_size=32,
)


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

model = tf.keras.Sequential([
    layers.Rescaling(1./255, input_shape=(256, 256, 3)),
    data_augmentation,
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.Dense(15, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.001), #lr can be changed
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


history = model.fit(
    data_train,
    validation_data=data_test,
    epochs=100
)


loss, accuracy = model.evaluate(data_test)
print("Test Accuracy:", accuracy)


plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.plot(history.history["accuracy"],label="accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

