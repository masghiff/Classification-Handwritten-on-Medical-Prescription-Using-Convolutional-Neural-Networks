import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
from tensorflow.keras import layers
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


transfer_learning_model = keras.models.Sequential() 
transfer_learning_model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(200,150,3)))
transfer_learning_model.add(layers.AveragePooling2D())

transfer_learning_model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
transfer_learning_model.add(layers.AveragePooling2D())
transfer_learning_model.add(layers.Flatten())

transfer_learning_model.add(layers.Dense(units=120, activation='relu'))

transfer_learning_model.add(layers.Dense(units=84, activation='relu'))

transfer_learning_model.add(layers.Dense(units=number_of_classes, activation = 'softmax'))

transfer_learning_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00007),
    loss='categorical_crossentropy',
    metrics=['accuracy',]
)

# Training Model

transfer_learning_model.fit(
    train_generator,
    epochs=100,
    verbose=1,
    validation_data=validation_generator,
)

# Save Model

transfer_learning_model.save( 'model_lenet100{datetime}.h5'.format(datetime=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")) )
transfer_learning_model.summary()

#predictions = transfer_learning_model.predict(validation_generator)# evaluate the model

#loss, accuracy, f1_score, precision, recall = transfer_learning_model.evaluate(Xtest, ytest, verbose=0)

#Y_pred = transfer_learning_model.predict_generator(validation_generator)
#y_pred = np.argmax(Y_pred, axis=1)
#print('Confusion Matrix')
#print(confusion_matrix(validation_generator.classes, y_pred))
#print("")
#print('Precision')
#print(precision_score(validation_generator.classes,y_pred,average=None))
#print("")
#print('Recall')
#print(recall_score(validation_generator.classes,y_pred,average=None))
#print("")
#print('f1_score')
#print(f1_score(validation_generator.classes,y_pred,average=None))
#print("")
#print("accuracy_score")
#print(accuracy_score(validation_generator.classes,y_pred))