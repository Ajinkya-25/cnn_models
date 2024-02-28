from tensorflow.keras.applications import MobileNetV3Large,MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,RandomContrast,RandomRotation,RandomFlip,Flatten
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import optimizers

data_train = image_dataset_from_directory(
    "F:\datasets\plants",
    validation_split=0.2,
    subset="training",
    shuffle=True,
    seed=42
)

data_test = image_dataset_from_directory(
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

#optional
augmentation = Sequential([
    RandomFlip(),
    RandomRotation(0.3),
    RandomContrast(0.2)
   # tf.keras.layers.AveragePooling2D()
])
aug_x_train = augmentation(x_train)


# Load pre-trained MobileNetV3 model without top layers (include_top=False)
base_model = MobileNetV3Large(weights='imagenet', include_top=False)

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(15, activation='softmax'))

# Freeze base model layers (optional)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=optimizers.Adamax(learning_rate=0.0009), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(aug_x_train, y_train, epochs=40, validation_data=(x_train,y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)