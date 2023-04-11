import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
from PIL import Image
from numpy import asarray
import numpy as np

im = Image.open("test2.jpg")
imResize = im.resize((150,200), Image.ANTIALIAS)
numpydata = asarray(imResize).tolist()
#print(numpydata.shape)

model = keras.models.load_model('modelusexception.h5')
print("load")

#ynew = model.predict_classes([numpydata])

predict_x=model.predict([numpydata])
classes_x=np.argmax(predict_x,axis=1)
cl=["Etagemycetin","Holidon","Ranitidine"]

print(cl[classes_x[0]])
#img_tensor = np.expand_dims(img_tensor, axis=0)