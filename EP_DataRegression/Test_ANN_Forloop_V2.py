import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import math
import time
import matplotlib.pyplot as plt


'''
inputs = keras.Input(shape=(784,), name="digits")
x1 = layers.Dense(64, activation="relu")(inputs)
x2 = layers.Dense(64, activation="relu")(x1)
outputs = layers.Dense(10, name="predictions")(x2)
model = keras.Model(inputs=inputs, outputs=outputs)
'''

def MyModelKeras():
  return tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    # Normalization_Layer,
    layers.Dense(units=1)
  ])

model = MyModelKeras()

# Instantiate an optimizer.
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = tf.keras.losses.MeanSquaredError()

# The actual line
TRUE_W = 3.0
TRUE_B = 2.0

NUM_EXAMPLES = 10000

# A vector of random x values
x = tf.linspace(-2,2, NUM_EXAMPLES)
x = tf.cast(x, tf.float32)

def f(x):
  return x * TRUE_W + TRUE_B

# Generate some noise
noise = tf.random.normal(shape=[NUM_EXAMPLES])

# Calculate y
y = f(x) + noise

# Prepare the dataset.
batch_size = 10

# User Input
train_percent = 0.7
test_percent = 0.15
val_percent = 0.15

x_data = np.reshape(np.array(x), (x.shape[0],1))
y_data = np.reshape(np.array(y), (y.shape[0],1))

# Creating Train Test Validation Set
x_train = x_data[0:math.floor(train_percent*x_data.shape[0]),0]
x_train = np.reshape(x_train,(x_train.shape[0],1))
y_train = y_data[0:math.floor(train_percent*y_data.shape[0]),0]
y_train = np.reshape(y_train,(y_train.shape[0],1))

train_index = x_train.shape[0]-1

x_test = x_data[train_index:train_index+math.floor(test_percent*x_data.shape[0]),0]
x_test = np.reshape(x_test,(x_test.shape[0],1))
y_test = y_data[train_index:train_index+math.floor(test_percent*y_data.shape[0]),0]
y_test = np.reshape(y_test,(y_test.shape[0],1))

test_index = x_test.shape[0]-1+train_index

x_val = x_data[test_index:test_index+math.floor(val_percent*x_data.shape[0]),0]
x_val = np.reshape(x_val,(x_val.shape[0],1))
y_val = y_data[test_index:test_index+math.floor(val_percent*y_data.shape[0]),0]
y_val = np.reshape(y_val,(y_val.shape[0],1))

# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# x_train = np.reshape(x_train, (-1, 784))
# x_test = np.reshape(x_test, (-1, 784))

#Reshaping x and y


'''
# Reserve 10,000 samples for validation.
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]
'''

buffer_input = 1000


# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=buffer_input).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)

'''
epochs = 2


for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            logits = model(x_batch_train, training=True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y_batch_train, logits)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %s samples" % ((step + 1) * batch_size))



# Get model
inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, name="predictions")(x)
model = keras.Model(inputs=inputs, outputs=outputs)


# Instantiate an optimizer to train the model.
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Prepare the metrics.
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
'''

# Prepare the metrics.
train_acc_metric = keras.metrics.MeanSquaredError()
val_acc_metric = keras.metrics.MeanSquaredError()

epochs = 3
'''
training_loss_value = tf.zeros([epochs])
val_loss_value = tf.zeros([epochs])
training_error_value = tf.zeros([epochs])
val_error_value = tf.zeros([epochs])
'''
training_loss_value_set = []
val_loss_value_set = []
training_error_value_set = []
val_error_value_set = []

# counter = -1

for epoch in range(epochs):
    # counter = counter + 1
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Update training metric.
        train_acc_metric.update_state(y_batch_train, logits)

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %d samples" % ((step + 1) * batch_size))


    #training_loss_value[counter] = loss_value
    training_loss_value_set = tf.experimental.numpy.append(training_loss_value_set, loss_value)

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    #training_error_value[counter] = train_acc
    training_error_value_set = tf.experimental.numpy.append(training_error_value_set, train_acc)

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        val_logits = model(x_batch_val, training=False)
        val_loss_value = loss_fn(y_batch_val, val_logits)
        # Update val metrics
        val_acc_metric.update_state(y_batch_val, val_logits)

    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))

    #val_loss_value[counter] = val_loss_value
    #val_error_value[counter] = val_acc

    val_loss_value_set = tf.experimental.numpy.append(val_loss_value_set, val_loss_value)
    val_error_value_set = tf.experimental.numpy.append(val_error_value_set, val_acc)



plt.figure()
plt.plot(training_loss_value_set, color='r', label='Train Loss')
plt.plot(val_loss_value_set, color='b', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss', labelpad=15)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure()
plt.plot(training_error_value_set, color='r', label='Train Error')
plt.plot(val_error_value_set, color='b', label='Validation Error')
plt.xlabel('Epoch')
plt.ylabel('Error', labelpad=15)
plt.legend()
plt.tight_layout()
plt.show()


Test_Y_Predict = np.zeros((x_test.shape[0]))

#LOOP: For Loop Simulation
for ii in range(x_test.shape[0]):
    Test_Y_Predict[ii] = model(x_test[ii])

plt.figure()
plt.plot(Test_Y_Predict, color='r', label='Predicted')
plt.plot(y_test, color='b', label='Actual', linestyle='dashed')
plt.xlabel('X')
plt.ylabel('Y', labelpad=15)
plt.legend()
plt.tight_layout()
plt.show()