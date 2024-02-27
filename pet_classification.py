import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

data_train = tf.keras.preprocessing.image_dataset_from_directory(
    "F:\datasets\catsdogs\kagglecatsanddogs_5340\PetImages",
    validation_split=0.2,
    subset="training",
    shuffle=True,
    seed=42
)

data_test = tf.keras.preprocessing.image_dataset_from_directory(
    "F:\datasets\catsdogs\kagglecatsanddogs_5340\PetImages",
    validation_split=0.2,
    subset="validation",
    shuffle=True,
    seed=42
)

'''label_names=['cat','dog']
plt.figure(figsize=(10, 10))
for image, labels in data_train.take(32):
    for i in range(32):
        plt.subplot(6, 6, i + 1)
        plt.imshow(image[i].numpy().astype("uint8"))  # Convert TensorFlow tensor to NumPy array
        plt.title(label_names[labels[i]])  # Convert TensorFlow tensor to NumPy array
        plt.axis("off")
plt.tight_layout()
plt.show()

'''
AUTOTUNE = tf.data.AUTOTUNE
train_ds = data_train.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = data_test.cache().prefetch(buffer_size=AUTOTUNE)

x_train, y_train = next(iter(data_train))
x_test, y_test = next(iter(data_test))

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.3),
])
aug_train=augmentation(x_train)

contrast=tf.keras.Sequential([
    layers.RandomContrast(0.8)
])
contrasted=contrast(x_train)

plt.figure(figsize=(10,10))
for i in range(32):
    plt.subplot(6, 6, i + 1)
    plt.imshow(contrasted[i].numpy().astype('uint8'))
    plt.axis('off')

plt.show()


# Iterate over each image in x_train


model=tf.keras.Sequential([

    layers.InputLayer(input_shape=(256,256,3)),
    layers.RandomContrast(0.5),
    layers.BatchNormalization(),
    layers.Conv2D(40,kernel_size=(5,5),activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(40,kernel_size=(3,3),activation='relu'),
    layers.MaxPool2D(),


    layers.Flatten(),
    layers.Dropout(0.4),
    layers.Dense(128,activation='relu'),
    layers.Dense(32,activation='relu'),
    layers.Dense(15,activation='sigmoid')

])



model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00013),metrics=['accuracy'],loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True))
history=model.fit(aug_train,y_train,epochs=25,validation_data=(x_test,y_test))


#loss ,acc=model.evaluate(x_test,y_test)
plt.plot(history.history['val_accuracy'])
plt.show()
