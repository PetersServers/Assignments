# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#print(tf.__version__)

'''first Tensorflow project '''
'''Image Classification'''

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



#explore the dataset

'''

print(train_images.shape)

print(train_labels)

print(len(train_labels))

print(test_images.shape)

print(len(test_labels))

print(train_images[0])


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

'''

#i break the information down to a scale between 0 and 1
#diminishes computational effort
#neural networks normally work between values of -1 and 1 in order
#to adjust values to work in multidimensional space

train_images = train_images / 255.0

test_images = test_images / 255.0

#print(test_images[0])
'''

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
'''

#image is now adjusted to index range of 1 from previous range of 255 bytes
# Each pixel typically consists of 8 bits (1 byte) for a Black and White (B&W) image
# or 24 bits (3 bytes) for a color image– one byte each for Red, Green, and Blue.
# 8 bits represents 28 = 256 tonal levels (0-255).

# That's why machine learning module that work with picture classification usually
# first use filters to reduce the colorscale of a picture --> normally converted to grey

#show first 20 pictures

'''

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[test_labels[i]])
plt.show()
'''
###

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
# the Flatten function of TF takes the information about the picture (array of 28*28 bytes)
# and converts it to a sequence o pixel data.
# the final result of this conversion is a list of 28*28 = 784 numbers in a list between 0 and 1
# thats also why the tf model chosen i sequential analysis of the byte information
    tf.keras.layers.Dense(128, activation='relu'), #why 128?
    tf.keras.layers.Dense(64),
    # i chose 128 because there is more complexity involved in recognizing clothes than in recognizing numbers
    # it splits the first layer into 128 different possible combinations of neurons that react to the information
    # of the 784 pixels/numbers
# the Dense function of TF converts the information to a neural network with 128 knots that are used
# to classify the datapoints that are inserted within multidimensional space
    tf.keras.layers.Dense(len(np.unique(test_labels))) #I also know that there are only 10 labels
    #the final layer than converts the 128 different possible neurons to the pssible output (in this case 10)

    #the number of neurons on each layer are should be the subject to experimentation

    #but this way it is more flexible
# the final layer denses them to the 10 main classes that there are in the dataset
])

model.compile(optimizer='adam',
# optimizer specifies the mathematical method that the model should implement during learning

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

# loss evaluates how accurate the model is during training

              metrics=['accuracy'])
# metrics are used to monitor the training and testing steps. The following example uses accuracy,
# the fraction of the images that are correctly classified.

#test the model x times and save the one that worked the best


'''
actual_best = 0
test_range = 10
for i in range(test_range):

    model.fit(train_images, train_labels, epochs=10)

    #fitts the training data to the training labels

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)

#to recieve feedback about the models accuracy after training

    if test_acc > actual_best:

        actual_best = test_acc
        print("better accuracy achieved")
        # Save the entire model as a SavedModel.
        model.save('saved_model/my_model')

#print(actual_best)

#classify the model that is best by reloading it after the test phase

'''


model = tf.keras.models.load_model('saved_model/my_model')
#print(model.summary())

'''make predicitions'''
#convert the output to predictions by adding a softmax layer
#it shows how likely the picture is to be related to the classes

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])


predictions = probability_model.predict(test_images)

#use the model to predict all the test images class
#f.e. take a look on the first image class that we predicted

#print(predictions[0])

#to see the highest value in the output (the class it most likely belongs to) use

#what are the first 20 pictures in the dataset

'''
for i in range(20):

    cls = np.argmax(predictions[i]) #prints class where the argument has the highest value
    print(class_names[cls])

print(actual_best)
'''

### plot how accurate the prediction of the picture was
#functions that plot when an argument is passed

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

'''

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()
'''
'''
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()
'''

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.

'''
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
'''
#predict a single picture

# Grab an image from the test dataset.

while True:

    chosen_image = int(input("choose an image number of the catalogue: "))

    img = test_images[chosen_image]

    print(img.shape)

    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img,0))

    print(img.shape)

    predictions_single = probability_model.predict(img)

    cls = np.argmax(predictions_single)  # prints class where the argument has the highest value
    print(class_names[cls])

    plt.figure()
    plt.imshow(test_images[chosen_image])
    plt.colorbar()
    plt.grid(False)
    plt.show()

'''genereal comments'''
#weighted product of matrix vector multiplication of one neuron layer points to the next neuron layer
# = (weighted function of values in rows multiplied by the sigmoid/value of the correlating neurons (- bias)
# bias = a value that is chosen as a safety measure so that only values above certain level are recognized as relevant
#point to the neurons of the next layer by squishing the output value of each layer to a number between 0 and 1

#today more ReLu is used instead of sigmoid because it is more effective than sigmoid
#ReLu is the idea of the neuron only lighting up if the fuction is abouve a certain threshhold,
#relative to the other result for the other different neurons
# --> is even closer to the biological idea of a neuron firing or not firing it's either a 0 or 1 relative to the
# other reults of the layer