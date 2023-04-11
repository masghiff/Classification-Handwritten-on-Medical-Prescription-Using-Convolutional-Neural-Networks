import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
image_target_size = (200, 150)
number_of_classes = 20

train_dir = 'resizedData/training'
val_dir = 'resizedData/validation'

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    fill_mode='nearest',
)

train_generator = train_datagen.flow_from_directory(  # YOUR CODE HERE
    train_dir,
    target_size=image_target_size,
    batch_size=32,
    class_mode='categorical',
)

validation_datagen = ImageDataGenerator(
    # YOUR CODE HERE)
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    fill_mode='nearest',
)

validation_generator = validation_datagen.flow_from_directory(  # YOUR CODE HERE
    val_dir,
    target_size=image_target_size,
    batch_size=32,
    class_mode='categorical',
)

# Fetching VGG16 Base Model

resnet_model = ResNet152(input_shape= image_target_size+(3,),
    include_top=False, weights='imagenet'
)

for layer in resnet_model.layers:
    layer.trainable = False

# Building Transfer Learning Model

transfer_learning_model = keras.models.Sequential([
    resnet_model,
    keras.layers.Flatten(),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(number_of_classes, activation='softmax'),
])

transfer_learning_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00007),
    loss='categorical_crossentropy',
    metrics=['accuracy',]
)

# Training Model

hist = transfer_learning_model.fit(
    train_generator,
    epochs=10,
    verbose=1,
    validation_data=validation_generator,
)

# Save Model

transfer_learning_model.save( 'resnet1puluh{datetime}.h5'.format(datetime=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")) )
transfer_learning_model.summary()


Y_pred = transfer_learning_model.predict_generator(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print("")
print('Precision')
print(precision_score(validation_generator.classes,y_pred,average=None))
print("")
print('Recall')
print(recall_score(validation_generator.classes,y_pred,average=None))
print("")
print('f1_score')
print(f1_score(validation_generator.classes,y_pred,average=None))
print("")
print("accuracy_score")
print(accuracy_score(validation_generator.classes,y_pred))
import matplotlib.pyplot as plt
plt.plot(hist.history["loss"])
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['val_accuracy'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()