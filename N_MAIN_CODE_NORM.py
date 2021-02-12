from __future__ import print_function
import sys
import matplotlib
import datetime
from tensorflow.examples.saved_model.integration_tests.mnist_util import INPUT_SHAPE
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras import layers
import tensorflow.keras as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.models as keras_models
from random import randrange
import Functions_RNN
import Generator_ads
import Memory_trace
import Models

Home_Address='.'
# List of Columns or inputs or features
Functions_RNN.Fill_sim_user_by_query_ad_keyword()
Functions_RNN.Fill_query_id_token_map()
Functions_RNN.Fill_Keyword_id_token_map()
# initialize the paths to our training and testing CSV files
TRAIN_CSV = Home_Address+'/track2/train/T_1.txt'
validation_CSV = Home_Address+'/track2/train/V.txt'
TEST_CSV = Home_Address+'/track2/train/TEST.txt'
SAVE_MODEL = '/home/mohammad/Project/models/'
ERROR_FILE = open(Home_Address+'/track2/ERROR_FILE.txt', "w")
Log_file = '/home/mohammad/Project/Log/'
Counter_file = '/home/mohammad/Project/Log/'
Model_file = '/home/mohammad/Project/Model/'
# initialize the number of epochs to train for and batch size
NUM_EPOCHS = 100
BS = 500

# labels in the dataset along with the testing labels
labels = set()
labels = {'0', '1'}
# output:NUM_TRAIN=151198702   labels={'0', '1'}
lb = LabelBinarizer()
lb.fit(list(labels))
print('Start Read From Training Dataset')

model = Models.Build_Model()

#****************Find Counter of which file is read
max_counter=0
try:
    counter_log = open(Counter_file + 'CounterLog.txt','r')
    for counter_line in counter_log:
        try:
            max_counter = int(counter_line.strip()) if int(counter_line.strip()) > max_counter else max_counter
        except:
            continue
    counter_log.close();
except:
    None
max_counter +=1
#****************************************

for i in range(max_counter, 301):
    #***************Check Memory
    '''
    local_vars = list(locals().items())
    sum_mem = 0
    max_mem = 0
    var_name_max_mem = ''
    for var, obj in local_vars:
        size_obj = sys.getsizeof(obj)
        sum_mem += size_obj
        if int(size_obj) > int(max_mem):
            max_mem = size_obj
            var_name_max_mem = var

    print('Sum of variable memory usage is: ' + str(sum_mem / (1024 * 1024)) + 'Mb' +
          ' **** ' + 'Maximume Varible is:' + var_name_max_mem + ':' + str(max_mem / (1024 * 1024)) + 'Mb')
    '''
    # ***************
    val_file_num = randrange(1,41)
    print('validation file number : '+str(val_file_num))
    print('Train file number : ' + str(i) )
    TRAIN_CSV = Home_Address+'/track2/train/T_'+str(i)+'.txt'
    validation_CSV = Home_Address+'/track2/train/V_'+str(val_file_num)+'.txt'
    trainGen = Generator_ads.Training_Generator(TRAIN_CSV, BS, lb, mode="train")
    validationGen = Generator_ads.Validation_Generator(validation_CSV, BS, lb, mode="train")

    try:
        model = keras_models.load_model(Model_file)
    except:
        print('Model is not built yet...')

    for j in range(1,3):
        f_log = open(Log_file + 'logfile.txt', 'a+')
        history=model.fit(x=trainGen, validation_data=validationGen, validation_steps=1, epochs=100, steps_per_epoch=1)
        f_log.write('Model is saved and i:' + str(i) +' **** j:'+str(j)+'  **** Validation File Number:   '+str(val_file_num)+'  **** Time:   '+str(datetime.datetime.now().ctime())+ '\n')
        f_log.close()

    model.save(Model_file, overwrite=True, include_optimizer=True)
    # Write counter in the file
    counter_log = open(Counter_file + 'CounterLog.txt', 'a+')
    counter_log.write(str(i)+'\n')
    counter_log.close()
    ###########################
    if i%10==0:
        break
    Memory_trace.Variable_mem()
    del model
    del trainGen
    del validationGen




