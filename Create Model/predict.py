import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
from PIL import Image
from numpy import asarray
import numpy as np

import os, sys
import cv2
from skimage.transform import resize
import keras.utils as image

print(keras.__version__)

imagex=cv2.imread("gambar/temp.jpg")
gray = cv2.cvtColor(imagex, cv2.COLOR_BGR2GRAY)
# Apply thresholding to obtain a binary image
threshold, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# Find the contours in the binary image
contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Iterate over the contours
ctr=0
biggestw=0

objsave=None
for cnt in contours:
    # Compute the bounding box of the contour
    x, y, w, h = cv2.boundingRect(cnt)
    luas=w*h
    if (luas>biggestw) :
        biggestw=luas
        # Extract the object from the image
        objsave = imagex[y:y+h, x:x+w]
cv2.imwrite('gambar/temp2.jpg', objsave)



imagex = cv2.imread("gambar/temp2.jpg")
gray = cv2.cvtColor(imagex, cv2.COLOR_BGR2GRAY)
median = cv2.medianBlur(gray,5)
dim = (250, 150)
resized = cv2.resize(median, dim, interpolation = cv2.INTER_AREA)
cv2.imwrite("gambar/temp3.jpg", resized)


im = image.load_img("gambar/temp3.jpg")

#img = image.load_img('image.jpg', target_size=(224, 224))

# Convert the image to a NumPy array
image_array = image.img_to_array(im)

# Resize the image
image_array = resize(image_array, (250, 150, 3))

#print(numpydata.shape)


model = keras.models.load_model('vggpre.h5')
#print("load")
#print(len(image_array))
#print(len(image_array[0]))
#print(len(image_array[0][0]))
#ynew = model.predict_classes([numpydata])

#print("predict")
img_test = np.expand_dims(image_array, axis=0)
predict_x=model.predict([img_test],verbose = 0)
#print("after")
#print(predict_x)
maxval=max(predict_x[0])
#print(maxval)
#for dtl in predict_x[0] :
#    print("detail")
#    print(dtl)
#print("maxval")
#print(maxval)
#print("finish")
classes_x=np.argmax(predict_x,axis=1)
cl=["B Complex","Bufagan","Citostan","Clorpramazin","Coparcetin","Etagemycetin","Etambutol","Fondazen","Haloperidol","Holidon","Lambucid","Orphen","Pyranzinamid","Ranitidine","Rifampicin","Scopmag Plus","Solpenox","Thryhexypenidyl","Trimakalk","Vibramox"]

print("######")
#print(classes_x)
if (maxval<0.8) :
    print("Tidak terdeteksi, class "+cl[classes_x[0]]+", confidence "+maxval)
else :
    print(cl[classes_x[0]])
#with open('class.txt', 'w') as f:
#    f.write(cl[classes_x[0]])
#img_tensor = np.expand_dims(img_tensor, axis=0)