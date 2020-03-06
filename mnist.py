from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt

# Normalizes the dataset
def normalize(data):
	data / 255

# Get pre-shuffled mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Visualize 1 data
plt.imshow(x_train[0])
plt.show()

print('PREPROCESSING DATA')
# (n, width, height) -> (n, depth, width, height)
# Images are grey scale so only 1 value to keep track of
#   therefore depth = 1
print('Reshaping the data...')
print('Training data\'s shape:', x_train.shape)
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
print('Reshaped Training data\'s shape:', x_train.shape)

# Normalize the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
normalize(x_train)
normalize(x_test)

print('PREPROCESSING LABELS')
# Convert 1D class labels into 2D class matrices
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

print('BUILDING MODEL')
# Build our model
model = Sequential()
# Add our first layer
# Parameters 1-3: # conv. filters, rows in each conv. kernel, # cols in each conv. kernel
model.add(Convolution2D(32, 3, activation = 'relu', input_shape = (1, 28, 28)))
# Add another layer
model.add(Convolution2D(32, 3, activation = 'relu'))
# Reduce # of parameters
model.add(MaxPooling2D(pool_size = (2,2)))
# Prevent overfitting
model.add(Dropout(0.25))
# Flatten the weights of the convolution layers
model.add(Flatten())
# Set output size of first layer
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
# Set output size of second layer (Final decisions)
model.add(Dense(10, activation = 'softmax'))
# Compile the model we just built
print('COMPILING MODEL')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# Fit the training data to the model just compiled
print('FITTING TRAINING DATA')
model.fit(x_train, y_train, batch_size = 32, nb_epoch = 10, verbose = 1)
# Test the model just trained
print('TESTING MODEL')
score = model.evaluate(x_test, y_test, verbose = 0)

print(score)