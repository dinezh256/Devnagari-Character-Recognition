# %%
'''
Importing Relevant Libaries
'''

# %%
# Standard useful data processing imports
import random
import numpy as np
import pandas as pd

# Visualisation imports
import matplotlib.pyplot as plt

# Scikit learn for preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Keras Imports - CNN
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

# Working directory
# Kaggle Terminal path, use your own local path
import os
os.chdir("../input")

# Loading the data
data = pd.read_csv("data.csv")
data.head(n=8)

data.groupby("character").count()


''' There seems to be no class imbalance in the dataset.
Let us visualize all the letters and numbers handwriiten in Devanagri. '''
char_names = data.character.unique()  
rows =10;columns=5;
fig, axis = plt.subplots(rows,columns, figsize=(8,16))
for row in range(rows):
    for col in range(columns):
        axis[row,col].set_axis_off()
        if columns*row+col < len(char_names):
            x = data[data.character==char_names[columns*row+col]].iloc[0,:-1].values.reshape(32,32)
            x = x.astype("float64")
            x/=255
            axis[row,col].imshow(x, cmap="binary")
            axis[row,col].set_title(char_names[columns*row+col].split("_")[-1])

plt.subplots_adjust(wspace=1, hspace=1)   
plt.show()


# Verifying the pixel distribution of any random character
import matplotlib.pyplot as plt
plt.hist(data.iloc[0,:-1])
plt.show()

''' Let us normalize all the pixel values and set the character column to target variable
Let's make the dependent(Y) and independent(X) datasets '''
X = data.values[:,:-1]/255.0
y = data["character"].values

#Let us minimize the RAM consumption
del data
n_classes = 46

# Let's split the data into train and test data, given test_size =0.15 in the research paper, but we have opted for 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Let's encode the categories
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_train = to_categorical(y_train, n_classes)
y_test = to_categorical(y_test, n_classes)


img_height_rows = 32
img_width_cols = 32
im_shape = (img_height_rows, img_width_cols, 1)
X_train = X_train.reshape(X_train.shape[0], *im_shape) #The ' * 'operator unpacks the tuple
X_test = X_test.reshape(X_test.shape[0], *im_shape)

# CNN Model - Sequential Modelling
cnn = Sequential()

kernelSize = (3, 3)
ip_activation = 'relu'
ip_conv_0 = Conv2D(filters=32, kernel_size=kernelSize, input_shape=im_shape, activation=ip_activation)
cnn.add(ip_conv_0)

# Add the next Convolutional+Activation layer
ip_conv_0_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)
cnn.add(ip_conv_0_1)

# Add the 1st Pooling layer
pool_0 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
cnn.add(pool_0)

# Add the 2D Convolution layers
ip_conv_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)
cnn.add(ip_conv_1)
ip_conv_1_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)
cnn.add(ip_conv_1_1)

# Add the 2nd Pooling layer
pool_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
cnn.add(pool_1)

# Let's deactivate around 20% of neurons randomly for training
drop_layer_0 = Dropout(0.2)
cnn.add(drop_layer_0)

flat_layer_0 = Flatten()
cnn.add(Flatten())

# Now add the Dense layers
h_dense_0 = Dense(units=128, activation=ip_activation, kernel_initializer='uniform')
cnn.add(h_dense_0)

# Let's add one more before proceeding to the output layer
h_dense_1 = Dense(units=64, activation=ip_activation, kernel_initializer='uniform')
cnn.add(h_dense_1)

op_activation = 'softmax'
output_layer = Dense(units=n_classes, activation=op_activation, kernel_initializer='uniform')
cnn.add(output_layer)

opt = 'adam'
loss = 'categorical_crossentropy'
metrics = ['accuracy']

# Compile the classifier using the configuration we want
cnn.compile(optimizer=opt, loss=loss, metrics=metrics)

# Print the layers used in the Neural Network part.
print(cnn.summary())


history = cnn.fit(X_train, y_train,
                  batch_size=200, epochs=50,
                  validation_data=(X_test, y_test))

scores = cnn.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100)) # Accuracy=98.67%

# Accuracy
print(history)
fig1, ax_acc = plt.subplots()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model - Accuracy')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()

''' According to the accuracy curves, the training and validation curves clearly follow the same trend throughout. 
This is not a case of overfitting or underfitting. '''

# Loss
fig2, ax_loss = plt.subplots()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model- Loss')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()