import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
import numpy as np
import os, sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from PIL import Image
from numpy import asarray
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

image_target_size = (200, 150)
number_of_classes = 20

train_dir = 'resizedData/training'
val_dir = 'resizedData/validation'
cl=["B Complex","Bufagan","Citostan","Clorpramazin","Coparcetin","Etagemycetin","Etambutol","Fondazen","Haloperidol","Holidon","Lambucid","Orphen","Pyranzinamid","Ranitidine","Rifampicin","Scopmag Plus","Solpenox","Thryhexypenidyl","Trimakalk","Vibramox"]


model = keras.models.load_model('resnet20.h5')
yreal=[]
ypred=[]
ctr=0
for item in cl :
    path=val_dir+"/"+item
    dirs = os.listdir(path)
    for item in dirs :
            fcheck=path+"/"+item
            if os.path.isfile(fcheck):
                print(fcheck)
                im = Image.open(fcheck)
                imResize = im.resize((150,200), Image.ANTIALIAS)
                numpydata = asarray(imResize).tolist()
                predict_x=model.predict([numpydata])
                classes_x=np.argmax(predict_x,axis=1)
                ypred.append(classes_x[0])
                yreal.append(ctr)
    ctr=ctr+1

#print("load")

#ynew = model.predict_classes([numpydata])
conf1=confusion_matrix(ypred,yreal)
#Y_pred = model.predict(validation_generator)
#y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(conf1)
print("")
print('Precision')
print(precision_score(ypred,yreal,average=None))
print("")
print('Recall')
print(recall_score(ypred,yreal,average=None))
print("")
print('f1_score')
print(f1_score(ypred,yreal,average=None))
print("")
print("accuracy_score")
print(accuracy_score(ypred,yreal))

print(classification_report(ypred, yreal))


#fig, ax = plot_confusion_matrix(conf_mat=conf1)
#plt.show()
#print(y_pred)
#print(validation_generator.classes)


cl2=[]
for itm in cl :
    cl2.append(itm[0:])
cmd = ConfusionMatrixDisplay(conf1, display_labels=cl2)
cmd.plot()
plt.show()

#for i in validation_generator:
#    idx = (validation_generator.batch_index - 1) * validation_generator.batch_size
#    print(validation_generator.filenames[idx : idx + validation_generator.batch_size])