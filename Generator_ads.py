import matplotlib
from reportlab.graphics import samples
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import Functions_RNN
import Memory_trace

#RNN
def Training_Generator(inputPath, bs: object, lb: object, mode: object = "train", aug: object = None) :
    # open the CSV file for reading
    f = open(inputPath, "r")
    while True:
        # initialize our batches of images and labels
        samples = []
        labels = []
        counter = 0
        # keep looping until we reach our batch size
        while counter < bs:
            # attempt to read the next line of the CSV file
            line = f.readline()
            # check to see if the line is empty, indicating we have
            # reached the end of the file
            if line == "":
                # reset the file pointer to the beginning of the file
                # and re-read the line
                f.seek(0)
                line = f.readline()

            # extract the label and construct the image
            line_error = line
            samples.append(Functions_RNN.Fill_List_Input(line))#Fill_List_Input_CR_Tokens
            line = line.strip().split(",")
            label = line[0]
            labels.append(label)
            counter = counter + 1
        # one-hot encode the labels
        labels = lb.transform(np.array(labels))
        yield np.array(samples, dtype=np.float32), labels
def Validation_Generator(inputPath, bs: object, lb: object, mode: object = "Test", aug: object = None) :
    # open the CSV file for reading
    f = open(inputPath, "r")
    while True:
        # initialize our batches of images and labels
        samples = []
        labels = []
        counter = 0
        # keep looping until we reach our batch size
        while counter < bs:
            # attempt to read the next line of the CSV file
            line = f.readline()
            # check to see if the line is empty, indicating we have
            # reached the end of the file
            if line == "":
                # reset the file pointer to the beginning of the file
                # and re-read the line
                f.seek(0)
                line = f.readline()

            # extract the label and construct the image
            line_error = line
            samples.append(Functions_RNN.Fill_List_Input(line))#Fill_List_Input
            line = line.strip().split(",")
            label = line[0]
            labels.append(label)
            counter = counter + 1
        # one-hot encode the labels
        labels = lb.transform(np.array(labels))
        yield np.array(samples, dtype=np.float32), labels
