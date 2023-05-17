import os
import re
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics 



#Data Loading

# Training Data
DIR_PATH = "./PatientData/"
patient_list = ['Patient_1','Patient_2', 'Patient_3','Patient_4']
train_img = list()
train_labels = list()
for patient in patient_list:
  curr_labellist = pd.read_csv(DIR_PATH+patient+'_Labels.csv')
  curr_path = DIR_PATH+patient+'/'
  filenames = os.listdir(curr_path)
  final_names = [i for i in filenames if re.search("(thresh.(?:jpg|gif|png))", i)]
  for i in range(len(curr_labellist)):
    ic_no = curr_labellist.loc[i,"IC"]
    ic_label = curr_labellist.loc[i,"Label"]
    filename = "IC_"+str(ic_no)+"_thresh.png"
    if filename in final_names:
      img = load_img(curr_path+filename, target_size=(224,224))
      img = img_to_array(img)
      train_img.append(img)
      if ic_label > 0:
        train_labels.append(1)
      else:
        train_labels.append(0)

train_img = np.array(train_img, dtype="float32")
train_labels = np.array(train_labels)


# Validation Data
DIR_PATH = "./PatientData/"
patient_list = ['Patient_5']
val_img = list()
val_labels = list()
for patient in patient_list:
  curr_labellist = pd.read_csv(DIR_PATH+patient+'_Labels.csv')
  curr_path = DIR_PATH+patient+'/'
  filenames = os.listdir(curr_path)
  final_names = [i for i in filenames if re.search("(thresh.(?:jpg|gif|png))", i)]
  for i in range(len(curr_labellist)):
    ic_no = curr_labellist.loc[i,"IC"]
    ic_label = curr_labellist.loc[i,"Label"]
    filename = "IC_"+str(ic_no)+"_thresh.png"
    if filename in final_names:
      img = load_img(curr_path+filename, target_size=(224,224))
      img = img_to_array(img)
      val_img.append(img)
      if ic_label > 0:
        val_labels.append(1)
      else:
        val_labels.append(0)

val_img = np.array(val_img, dtype="float32")
val_labels = np.array(val_labels)

# Building the VGG16 model

backbone = applications.VGG16(weights = "imagenet", input_shape=(224,224,3), include_top = False)
backbone.trainable = False
flat = layers.Flatten()(backbone.output)

cls1 = layers.Dense(units=512, activation = 'relu')(flat)
dropout1 = layers.Dropout(rate = 0.5)(cls1)
cls2 = layers.Dense(units=512, activation = 'relu')(dropout1)
dropout2 = layers.Dropout(rate = 0.5)(cls2)
clsoutput = layers.Dense(units=2, activation = 'softmax', name="clshead")(dropout2)

model = models.Model(
    inputs=backbone.input,
    outputs=(clsoutput))

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4), 
    loss=losses.sparse_categorical_crossentropy,
    loss_weights={'clshead': 1.0},
    metrics=['acc'], 
)

model.summary()

# Training the model

history = model.fit(
    train_img, {'clshead': train_labels},
    validation_data=(val_img, {'clshead': val_labels}),
    batch_size=32,
    epochs=21,
    verbose=1)

models.save_model(model, "./trained_model.h5")


