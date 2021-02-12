
from __future__ import print_function
import sys
import matplotlib
from tensorflow.python.ops.while_v2 import _is_trainable
matplotlib.use('Agg')
matplotlib.use('TkAgg')
import matplotlib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import tensorflow.keras.models as keras_models
from random import randrange
import Functions_RNN
import Generator_ads
import Memory_trace
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import pandas as pd

Functions_RNN.Fill_sim_user_by_query_ad()
Functions_RNN.Fill_query_id_token_map()

Home_Address='.'
TEST_CSV = Home_Address+'track2/train/T_30.txt'
Model_file = Home_Address+'Log/Model'
f_test = open(TEST_CSV, "r")
y_test = []
count_line=100000
lines=[]
counter=0
for line in f_test:
    lines.append(line)
    line = line.strip().split(',')
    y_test.append(line[0])
    counter+=1
    if counter == count_line:
        break
f_test.close()
labels = set()
labels = {'0', '1'}
model = keras_models.load_model(Model_file)
lb = LabelBinarizer()
lb.fit(list(labels))
testGen = Generator_ads.Training_Generator(TEST_CSV, 2000, lb, mode="train")
y_pred_keras = model.predict(x=testGen, steps=50)
y_hat=[round(item,10) for ite in y_pred_keras for item in ite]
df=pd.DataFrame({'line':lines,'y':y_test,'yhat':y_hat})
df=df.sort_values(by=['yhat','y'],ascending=False)

val=roc_auc_score(y_test, check)
print(str(val))
