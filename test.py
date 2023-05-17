import os
import re
import numpy as np
import pandas as pd
import csv

from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

from sklearn.metrics import confusion_matrix

TEST_DIR_PATH = "./testPatient/"
test_patient = ['test_Data']

filenames1 = os.listdir(TEST_DIR_PATH)
final_names_csv = [i for i in filenames1 if re.search("(.(?:csv))", i)]
csv_path = final_names_csv[0]
# print(TEST_DIR_PATH+csv_path)

loaded_model = models.load_model("./trained_model.h5")

test_img = list()
test_labels = list()
# pred_labels = list()
pred_list = list()
for patient in test_patient:
    final_out_pred = list()
    curr_labellist = pd.read_csv(TEST_DIR_PATH+csv_path)
    # print(curr_labellist)
    curr_path = TEST_DIR_PATH+patient+'/'
    filenames = os.listdir(curr_path)
    final_names = [i for i in filenames if re.search("(thresh.(?:jpg|gif|png))", i)]
    for i in range(len(curr_labellist)):
        temp_out_pred = []
        ic_no = curr_labellist.loc[i,"IC"]
        ic_label = curr_labellist.loc[i,"Label"]
        # print(ic_label)
        filename = "IC_"+str(ic_no)+"_thresh.png"
        if filename in final_names:
            
            img = load_img(curr_path+filename, target_size=(224,224))
            img = img_to_array(img)

            img = np.expand_dims(img, axis=0)
            pred_label = loaded_model.predict(img)
            temp_out_pred.append(ic_no)
            if pred_label[0][0] > pred_label[0][1]:
                temp_out_pred.append(0)
                pred_list.append(0)
            else:
                temp_out_pred.append(1)
                pred_list.append(1)

            if ic_label > 0:
                test_labels.append(1)
            else:
                test_labels.append(0)
        final_out_pred.append(temp_out_pred)

# test_img = np.array(test_img, dtype="float32")
test_labels = np.array(test_labels)
# print(test_labels)

with open('./Results.csv', 'w') as file1:
  heading = ['IC_Number', 'Label']
  write = csv.writer(file1)
  write.writerow(heading)
  write.writerows(final_out_pred)



cm1 = confusion_matrix(test_labels, pred_list)

print(cm1)
total1=sum(sum(cm1))
accuracy1=(cm1[0,0]+cm1[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)

precision = cm1[0,0]/(cm1[0,0]+cm1[1,0])
print('Precision :', precision)

metrics_list = [['Accuracy', str(accuracy1*100)+'%'],
                ['Precision', str(precision*100)+'%'],
                ['Sensitivity', str(sensitivity1*100)+'%'],
                ['Specificity', str(specificity1*100)+'%']
]

with open('./Metrics.csv', 'w') as file2:
#   heading = ['IC_No', 'Label']
  write = csv.writer(file2)
#   write.writerow(heading)
  write.writerows(metrics_list)

