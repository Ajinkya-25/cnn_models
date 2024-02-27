#import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l

data_train = tf.keras.preprocessing.image_dataset_from_directory(
    "F:\datasets\plants",
    validation_split=0.2,
    subset="training",
    shuffle=True,
    seed=42
)

data_test = tf.keras.preprocessing.image_dataset_from_directory(
    "F:\datasets\plants",
    validation_split=0.2,
    subset="validation",
    shuffle=True,
    seed=42
)



# Applying normalization and shuffling to training data
train_ds = data_train.shuffle(1000)
x_train, y_train = next(iter(train_ds))

# Normalizing the pixel values of test data
test_ds = data_test
x_test, y_test = next(iter(test_ds))

# Data augmentation
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.3),
   # tf.keras.layers.AveragePooling2D()
])

aug_x_train = augmentation(x_train)
#aug_x_train = aug1(aug_x_train//4)
# Training the model



def AlexNet(lr=0.01, num_classes=15):
    net = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4,
                                   activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same',
                                   activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                                   activation='relu'),
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                                   activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                                   activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='softmax'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes)])
    return net


with d2l.try_gpu():
    model = AlexNet(lr=0.001)

    # Compile the model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'],
                  optimizer='adam')

    # Train the model using the training dataset
model.fit(data_train, epochs=20,validation_data=(x_test,y_test))

    # Evaluate the model using the test dataset
loss, acc = model.evaluate(data_test)
print(acc)

