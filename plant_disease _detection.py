import tensorflow as tf
from tensorflow.keras import layers
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


train_ds = data_train
x_train, y_train = next(iter(train_ds))

# Normalizing the pixel values of test data
test_ds = data_test
x_test, y_test = next(iter(test_ds))

# Data augmentation
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomContrast(0.2)
   # tf.keras.layers.AveragePooling2D()
])

aug_x_train = augmentation(x_train)

model=tf.keras.Sequential([

    layers.InputLayer(input_shape=(256,256,3)),
    layers.RandomContrast(0.5),
    layers.BatchNormalization(),
    layers.Conv2D(32,kernel_size=(7,7)),
    layers.MaxPool2D(),
   # layers.Conv2D(64,kernel_size=(5,5)),
    #layers.MaxPool2D(),
    layers.Conv2D(128,kernel_size=(3,3),activation='relu'),
    layers.MaxPool2D(),
    #layers.Conv2D(128,kernel_size=(3,3),activation='relu'),
    #layers.MaxPool2D(),

    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(128,activation='relu'),
    layers.Dense(32,activation='relu'),
    layers.Dense(15,activation='relu')


])


model.compile(optimizer=tf.keras.optimizers.Adamax(lr=0.00013),metrics=['accuracy'],loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(aug_x_train,y_train,epochs=20,validation_data=(x_test,y_test))
#loss ,acc=model.evaluate(x_test,y_test)
