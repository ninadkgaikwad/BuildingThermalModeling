import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()

dataset.isna().sum()

dataset = dataset.dropna()

dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})

dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
dataset.tail()

# Work from here----under for loop
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

plt.figure()
sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
plt.savefig('pic0.png')
plt.show()


train_dataset.describe().transpose() #no need

train_features = train_dataset.copy() # X Value
test_features = test_dataset.copy() # X Value

train_labels = train_features.pop('MPG') # Y Value
test_labels = test_features.pop('MPG') # Y Value

train_dataset.describe().transpose()[['mean', 'std']] #no need

normalizer = tf.keras.layers.Normalization(axis=-1)#no need

normalizer.adapt(np.array(train_features))#no need

print(normalizer.mean.numpy()) #no need

'''
first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())
'''

horsepower = np.array(train_features['Horsepower']) #new x

horsepower_normalizer = layers.Normalization(input_shape=[1,], axis=None)
horsepower_normalizer.adapt(horsepower)

horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])

horsepower_model.summary()#no need

horsepower_model.predict(horsepower[:10])#no need

horsepower_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_squared_error')
# learning rate , loss, epochs, verbose, validation_split goes to  user input
# loss = 'mean_absolute_error'

history = horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail() #no need

horsepower_model.save('model1')
horsepower_model = keras.models.load_model('model1')#no need


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.ylim([0, 10])
plt.xlabel('Epoch')
plt.ylabel('Error [MPG]') #change mpg to y column name
plt.legend()
plt.grid(True)
plt.savefig('pic1.png') #dynamic name,create regression folder in the result (create new folder)
plt.show()

test_results = {}

test_results['horsepower_model'] = horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels, verbose=0)

#test_results[key], features = testx, lables = test y

x = tf.linspace(0.0, 250, 251) #x = x values of the test data
y = horsepower_model.predict(x)


plt.scatter(train_features['Horsepower'], train_labels, label='Data')
plt.plot(x, y, color='k', label='Predictions')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.legend()
plt.savefig('pic2.png')
plt.show()


