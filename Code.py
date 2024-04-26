#ImprovEYES: deep-learning CNN pipeline to diagnose diabetic retinopathy
​
import numpy as np
#Assorts data into matrices for mathematical analyzation
import pandas as pd
#Analayzes labeled data
import os
#Facilitates interactions with operating system 
import pathlib
#Manipulating files 
import PIL
#Opening/saving images
import cv2
#Computer vision
import matplotlib.pyplot as plt
import seaborn as sns
#Used to plot graphs about model status
sns.set_style('whitegrid')
import random
#Random mini-batches for SGD
import itertools
#Iterations/epochs for model improvement
import tensorflow as tf
#Model improvement after peformance gauging
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#Data augmentation
from tensorflow.keras.applications import EfficientNetB3
#Base layer of ImprovEYES, convolves with input images
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
#Creating layers
from tensorflow.keras.layers import InputLayer, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten, Dense, Activation
#Creating layers, implementing regularization techniques
from tensorflow.keras.metrics import categorical_crossentropy
#Based on linear regression, cost function to calculate loss
from tensorflow.keras.optimizers import Adam, Adamax
#Variant of SGD to reduce loss
from tensorflow.keras import regularizers
#Provides L1 & L2 functions
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
#Validation loss, stops training if epochs don't improve
from PIL import Image, UnidentifiedImageError
#Image: resizes & crops, UIE: handles errors PIL can't identify
from sklearn.preprocessing import LabelEncoder
#Converts strings to integers for numerical arrays
from sklearn.model_selection import train_test_split
#Splits testing & training datasets
from sklearn.metrics import confusion_matrix, classification_report
#Chart demonstrating false positive/negatives
import warnings
warnings.filterwarnings("ignore")
#Eliminating minor warnings for cleaner output
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
#Suppressing loquacious messages for cleaner output
print("ImprovEYES is ready to analyze and Improv your EYES.")
Path_data = '/kaggle/input/diabetic-retinopathy-dataset'
#Linking the training data 
data = os.listdir(Path_data)
Healthy = os.listdir('/kaggle/input/diabetic-retinopathy-dataset/Healthy')
MildNPDR = os.listdir('/kaggle/input/diabetic-retinopathy-dataset/Mild DR')
ModerateNPDR = os.listdir('/kaggle/input/diabetic-retinopathy-dataset/Moderate DR')
ProliferativeDR = os.listdir('/kaggle/input/diabetic-retinopathy-dataset/Proliferate DR')
SevereDR = os.listdir('/kaggle/input/diabetic-retinopathy-dataset/Severe DR')
#Referencing each folder in the dataset
print("ImprovEYES Analysis can fall into the following categories:", (data), "\n______________________________\n")
print("Number of classes :", len(data), "\n______________________________\n")
print("Number of Healty images :", len(Healthy), "\n______________________________\n")
print("Number of Mild NDPR images :", len(MildNPDR),  "\n______________________________\n")
print("Number of Moderate NDPR images :", len(ModerateNPDR),  "\n______________________________\n")
print("Number of Proliferative DR images :", len(ProliferativeDR),  "\n______________________________\n")
print("Number of Severe DR images :", len(SevereDR),  "\n______________________________\n")
imgpaths = []
labels =[]
#Stores images and their categories (labels)
data = os.listdir(Path_data)
#Retrieves necessary files
​
for i in data:
    classpath = os.path.join(Path_data, i)
    imgpaths.extend([os.path.join(classpath, img) for img in os.listdir(classpath)])
    labels.extend([i] * len(os.listdir(classpath)))
#for i in data:
#    classpath = os.path.join(Path_data, i)
#    imglist = os.listdir(classpath)
#    for img in imglist:
#        imgpath = os.path.join(classpath, img)
#        imgpaths.append(imgpath)
#        labels.append(i)
Paths = pd.Series(imgpaths, name = 'Paths')
print(Paths)
Labels = pd.Series(labels, name = 'Labels')
#Creates two pandas series: paths (images) & labels (cateogries)
#Classpath iterates over every pixel in the image
#Imglist iterates over every picture collected from classpath
​
Df= pd.concat([Paths, Labels], axis = 1)
Df.head(5)
#Combining paths & labels to form a DF; printing first 5 rows
​
train, testval = train_test_split(Df, test_size = 0.2, shuffle = True, random_state = 123)
valid, test = train_test_split(testval, test_size = 0.5, shuffle = True, random_state = 123)
#Splitting DF into three subsets: train (80%), validate (10%), test (10%)
​
#print("Train shape: ", train.shape)
#print("Valid shape: ", valid.shape)
#print("Test shape: ",test.shape)
​
train.Labels.value_counts()
#Counting each unique example during training
​
batch_size = 20
#Smaller batch sizes to train data for more data generation
img_size = (224, 224)
#Standard values
channels = 3
#RGB
img_shape = (img_size[0], img_size[1], channels)
#Specifies height/width
​
tr_G = ImageDataGenerator(
    zca_whitening=True,
    rotation_range=30.,
    fill_mode='nearest',
    )
​
#Keras library, decorrelates pixels, randomly rotates 30+-, fills empty pixels
​
V_G = ImageDataGenerator()
t_G = ImageDataGenerator()
#Testing and training
​
Train = tr_G.flow_from_dataframe(train, x_col = 'Paths', y_col = 'Labels', target_size = img_size, class_mode = 'categorical', color_mode = 'rgb', shuffle = True, batch_size = batch_size)
Valid = V_G.flow_from_dataframe(valid, x_col = 'Paths', y_col = 'Labels', target_size = img_size, class_mode = 'categorical', color_mode = 'rgb', shuffle = True, batch_size = batch_size)
Test = t_G.flow_from_dataframe(test, x_col = 'Paths', y_col = 'Labels', target_size = img_size, class_mode = 'categorical', color_mode = 'rgb', shuffle = False, batch_size = batch_size)
#Data augmentation for different batches
​
L_index = Train.class_indices
#L_index
#Assigning categories that are easy for the algorithm to understand
​
Keys = list(L_index.keys())
#Keys
imgs, labels = next(Train)
#Inspecting sample batches
​
plt.figure(figsize= (15, 15))
#Plotting a blank canvas using matplotlib
​
for i in range(8):
    plt.subplot(3, 4, i +1)
    im = imgs[i]/255
    plt.imshow(im)
#Subplots in the previously made blank canvas for easy analysis
    index = np.argmax(labels[i])
    label = Keys[index]
    plt.title(label, color = 'purple')
    plt.axis('off')
#Adjusts axises, assigns subplots to images
​
plt.tight_layout()    
plt.show()
#Adjusts spaces, shows the image itself
​
n_classes = len(list(Train.class_indices.keys()))
n_classes
#Defines number of classes 
​
img_shape=(img_size[0], img_size[1], 3)
model_name='EfficientNetB3'
base_model= EfficientNetB3(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max') 
#Resizing image, processing picture through Keras library
​
base_model.trainable=True
x=base_model.output
x=BatchNormalization(axis=-1, momentum=0.999, epsilon=0.001 )(x)
x = Dense(1024, kernel_regularizer = regularizers.l2(l = 0.01),activity_regularizer=regularizers.l1(0.005),
                bias_regularizer=regularizers.l1(0.005) ,activation='relu')(x)
x=Dropout(rate=.2, seed=123)(x)
x = Dense(512, kernel_regularizer = regularizers.l2(l = 0.01),activity_regularizer=regularizers.l1(0.005),
                bias_regularizer=regularizers.l1(0.005) ,activation='relu')(x)
x=Dropout(rate=.3, seed=123)(x)
x = Dense(256, kernel_regularizer = regularizers.l2(l = 0.01),activity_regularizer=regularizers.l1(0.005),
                bias_regularizer=regularizers.l1(0.005) ,activation='relu')(x)
x=Dropout(rate=.4, seed=123)(x)
output=Dense(n_classes, activation='softmax')(x)
model=Model(inputs=base_model.input, outputs=output)
lr=.0001
#Extends EfficientNetB3
#Updates weights in the CF 
#Small LR = small steps, slower learning but higher accuracy 
​
model=Sequential()
model.add(base_model)
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(512, activation='elu'))
model.add(Dense(256, activation='elu'))
model.add(Dense(128, activation = 'elu'))
model.add(Dense(5, activation='softmax'))
model.compile(
    Adamax(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['acc']
)
model.summary()
#Flatterns the input into a 1D tensor
#Prevents the data from become too specialized
#More layers/neurons added
​
from tensorflow.keras.utils import plot_model
from IPython.display import Image
plot_model(model, to_file='convnet.png', show_shapes=True,show_layer_names=True)
Image(filename='convnet.png')
#Mimics Jupyter Notebook by showing the image
​
epochs = 50
#Number of iterations
history = model.fit(x= Train, epochs= epochs, verbose= 1, validation_data= Valid, validation_steps= None, shuffle= False)
#Trains model on 50 epochs, optionally validation set
​
tr_acc = history.history['acc']
tr_loss = history.history['loss']
#Training model isn't always fully accurate; computes loss/accuracy
​
v_acc = history.history['val_acc']
v_loss = history.history['val_loss']
#Validating accuracy of unseen data
​
index_acc = np.argmax(v_acc)
high_Vacc = v_acc[index_acc]
#Finds the best accuracy of the validation set 
​
index_loss = np.argmin(v_loss)
low_Vloss = v_loss[index_loss]
#Find the lowest accuracy of validation set
​
Epochs =[]
for i in range(len(tr_acc)):
    Epochs.append (i+1)
    
best_acc = f'Best epoch ={str(index_acc +1)}'
best_loss = f'Best epoch ={str(index_loss+1)}'
#Defining best epoch
​
plt.figure(figsize = (16, 8))
plt.style.use('fivethirtyeight')
#Configuring size of shape
​
plt.subplot(1,2,1)
plt.plot(Epochs, tr_acc, "g", label = "Train Accuarcy")
plt.plot(Epochs, v_acc, "r", label = "Valid Accuarcy")
plt.scatter(index_acc+1, high_Vacc, s= 150, color = 'purple', label = best_acc)
#Subplot with two simulatenous sides: marking epochs and recording best one
​
plt.title("Accuracy: Train Vs valid")
plt. xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
#A plot with its titles: acuracy, axises: epochs and accuracy
​
plt.subplot(1,2,2)
plt.plot(Epochs, tr_loss, "g", label = "Train Loss")
plt.plot(Epochs, v_loss, "r", label = "Valid Loss")
plt.scatter(index_loss+1, low_Vloss, s= 150, color = 'purple', label = best_loss)
#Two subplots: one about training loss and validation loss over epochs
#Epoch with lowest validation loss has a purple scatter plot
​
plt.title("Loss: Train Vs valid")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#A plot with its titles: acuracy, axises: epochs and accuracy
​
plt.tight_layout()
plt.show()
#Formatting and displaying the plot
​
Train_sc = model.evaluate(Train, verbose = 1)
Valid_sc = model.evaluate(Valid, verbose = 1)
Test_sc =model.evaluate(Test, verbose = 1)
#Evaluates the model across each of its datasets
​
#Print
#print('Train Scores : \n    accuracy:', Train_sc[1], '\n      Loss: ', Train_sc[0], '\n________________________')
#print('Valid Scores : \n    accuracy:', Valid_sc[1], '\n      Loss: ', Valid_sc[0], '\n________________________')
#print('Test Scores : \n    accuracy:', Test_sc[1], '\n      Loss: ', Test_sc[0], '\n________________________')
​
predictions = model.predict_generator(Test)
y_pred = np.argmax(predictions, axis = 1)
#Extracing predicted values
​
#Chack
#print(predictions)
#print(y_pred)
​
# Use n. of keys of  Class indices to greate confusion matrix
Test_cl_ind = Test.class_indices
#Retrieves class indices (categories)
​
classes = list(Test_cl_ind.keys())
#Obtaining classes
​
cm = confusion_matrix(Test.classes, y_pred)
cm
#CMC (Analyzing where the algorithm makes mistakes)
​
plt.figure(figsize =(8, 8))
plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Greens)
plt.title("Confusion Matrix")
plt.colorbar()
#Demonstrating performance of a classification model 
​
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes,rotation = 45)
plt.yticks(tick_marks, classes)
#Customize the plot, optionally rotates for readability
​
thresh = cm.max()/2
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(i, j, cm[i, j], horizontalalignment = 'center', color = 'white' if cm[i,j] > thresh  else 'red')
#Changing color based on importance of data    
plt.tight_layout()
plt.xlabel('Predictions')
plt.ylabel('Real Values')
plt.show()
#Adjusts layout of computer matrix
​
#model.save('effB3 CNN DR.h5')
