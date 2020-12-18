# %% Import libraries
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from tqdm import tqdm
import tensorflow_datasets as tfds

# %% Gpu Setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# %% Load and Preprocess the Dataset
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
data_file = tf.keras.utils.get_file("breast_cancer.csv", DATASET_URL)
col_names = ["id", "clump_thickness", "un_cell_size", "un_cell_shape", "marginal_adheshion", "single_eph_cell_size", "bare_nuclei", "bland_chromatin", "normal_nucleoli", "mitoses", "class"]
df = pd.read_csv(data_file, names=col_names, header=None)

df.info()
"""
 #   Column                Non-Null Count  Dtype 
---  ------                --------------  ----- 
 0   id                    699 non-null    int64 
 1   clump_thickness       699 non-null    int64 
 2   un_cell_size          699 non-null    int64 
 3   un_cell_shape         699 non-null    int64 
 4   marginal_adheshion    699 non-null    int64 
 5   single_eph_cell_size  699 non-null    int64 
 6   bare_nuclei           699 non-null    object
 7   bland_chromatin       699 non-null    int64 
 8   normal_nucleoli       699 non-null    int64 
 9   mitoses               699 non-null    int64 
 10  class                 699 non-null    int64
 """
df.drop(['id'], axis = 1, inplace = True)

 
import seaborn as sns

for each in df.columns:
    sns.displot(df[each])
    plt.savefig(str(each) + '.png')
# Classes are only 2.0 and 4.0

df['class'].value_counts()
"""
2    458
4    241
Name: class, dtype: int64
"""
 
df['bare_nuclei'].value_counts()
"""
1     402
10    132
5      30
2      30
3      28
8      21
4      19
?      16
9       9
7       8
6       4
Name: bare_nuclei, dtype: int64
"""

df['bare_nuclei'] = df['bare_nuclei'].replace('?', np.nan)
df[['bare_nuclei']] = df[['bare_nuclei']].astype('float') 

df['bare_nuclei'] = df['bare_nuclei'].replace(np.nan, np.mean(df['bare_nuclei'])) 
 
df['class'] = [0 if each == 2 else 1 for each in df['class']] # binary classification


# Train - Test
test_size = 0.21
random_state = 13

labels = df.iloc[:, 9]
df.drop(['class'], axis = 1, inplace = True)

x_train,x_test, y_train, y_test = train_test_split(df, labels, test_size = test_size, random_state = random_state)

train_stats = x_train.describe()
train_stats = train_stats.transpose() 

# Normalize 
x_train = (x_train - np.mean(x_train)) / (np.std(x_train))
x_test = (x_test - np.mean(x_test)) / (np.std(x_test))
 
# Create TF datasets to manage the pipeline much easier
train_dataset = tf.data.Dataset.from_tensor_slices((x_train.values, y_train.values))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test.values, y_test.values))
 
batch_size = 32
train_dataset = train_dataset.shuffle(buffer_size=len(x_train)).batch(batch_size)

test_dataset =  test_dataset.batch(batch_size=batch_size) 
 
# %% Define the Model 
 
from tensorflow.keras import backend as K
def my_relu(x):
    return K.maximum(-0.21, x)

def base_model():
    inputs = tf.keras.layers.Input(shape = (len(x_train.columns)))
    x = tf.keras.layers.Dense(192, activation= my_relu, name = 'first_dense')(inputs)
    x = tf.keras.layers.Dense(64, activation=my_relu, name = 'second_dense')(x)
    out_layer = tf.keras.layers.Dense(1, activation= 'sigmoid', name = 'last_layer')(x)
    model = tf.keras.Model(inputs = inputs, outputs = out_layer)
    return model

model = base_model()

from tensorflow.python.keras.utils.vis_utils import plot_model
plot_model(model,show_shapes = True, show_layer_names=True, to_file = 'model.png')

    
# %% Define Optimizer and Loss Objects
learning_rate = 0.001 # easy for tweaking

optimizer = tf.keras.optimizers.Adam(lr = learning_rate)
loss_object = tf.keras.losses.BinaryCrossentropy()

# %% Evaluate Un-Trained Model to see the loss
outputs = model(x_test.values)
loss_value = loss_object(y_true=y_test.values.reshape(-1,1), y_pred=outputs)
print("Loss before training %.5f" % loss_value.numpy()) # Loss before training 0.84200

def plot_confusion_matrix(y_true, y_pred, title='', labels=[0,1]):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title(title)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
          plt.text(j, i, format(cm[i, j], fmt),
                  horizontalalignment="center",
                  color="black" if cm[i, j] > thresh else "white")
    plt.show()

plot_confusion_matrix(y_test.values, tf.round(outputs), title='Confusion Matrix for Untrained Model')

# %% Create F1 Metric for better metric

class F1Score(tf.keras.metrics.Metric):

    def __init__(self, name='f1_score', **kwargs):       
        # call the parent class init
        super(F1Score, self).__init__(name=name, **kwargs)

        # Initialize Required variables
        # true positives
        self.tp = tf.Variable(0, dtype = 'int32')
        # false positives
        self.fp = tf.Variable(0, dtype = 'int32')
        # true negatives
        self.tn = tf.Variable(0, dtype = 'int32')
        # false negatives
        self.fn = tf.Variable(0, dtype = 'int32')

    def update_state(self, y_true, y_pred, sample_weight=None):

        # Calculate confusion matrix.
        conf_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=2)
        
        # Update values of true positives, true negatives, false positives and false negatives from confusion matrix.
        self.tn.assign_add(conf_matrix[0][0])
        self.tp.assign_add(conf_matrix[1][1])
        self.fp.assign_add(conf_matrix[0][1])
        self.fn.assign_add(conf_matrix[1][0])

    def result(self):
        '''Computes and returns the metric value tensor.'''

        # Calculate precision
        if (self.tp + self.fp == 0):
            precision = 1.0
        else:
            precision = self.tp / (self.tp + self.fp)
      
        # Calculate recall
        if (self.tp + self.fn == 0):
            recall = 1.0
        else:
            recall = self.tp / (self.tp + self.fn)

        f1_score = 2 * ((precision * recall) / (precision + recall))
 
        return f1_score

    def reset_states(self):
        '''Resets all of the metric state variables.'''
        # The state of the metric will be reset at the start of each epoch.
        self.tp.assign(0)
        self.tn.assign(0) 
        self.fp.assign(0)
        self.fn.assign(0)


train_f1score_metric = F1Score()
val_f1score_metric = F1Score()

train_acc_metric = tf.keras.metrics.BinaryAccuracy()
val_acc_metric = tf.keras.metrics.BinaryAccuracy()

# %% Apply Gradients

def apply_gradient(optimizer, loss_object, model, x, y):   
    with tf.GradientTape() as t:
        logits = model(x)
        loss_value = loss_object(y , logits)
  
    gradients = t.gradient(loss_value , model.trainable_weights)
    optimizer.apply_gradients(zip(gradients , model.trainable_weights))
  
    return logits, loss_value

# %% Train for One Epoch
def train_data_for_one_epoch(train_dataset, optimizer, loss_object, model, 
                             train_acc_metric, train_f1score_metric, verbose=True):

    losses = []

    #Iterate through all batches of training data
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        #Calculate loss and update trainable variables using optimizer
        logits, loss_value = apply_gradient(optimizer, loss_object , model , x_batch_train , y_batch_train)
        losses.append(loss_value)

        #Round off logits to nearest integer and cast to integer for calulating metrics
        logits = tf.round(logits)
        logits = tf.cast(logits, 'int64')

        #Update the training metrics
        train_acc_metric.update_state(y_batch_train, logits)
        train_f1score_metric.update_state(y_batch_train, logits)

        #Update progress
        if verbose:
            print("Training loss for step %s: %.4f" % (int(step), float(loss_value)))
    
    return losses

# %% Perform Validation
def perform_validation():
    losses = []

    #Iterate through all batches of validation data.
    for x_val, y_val in test_dataset:

        #Calculate validation loss for current batch.
        val_logits = model(x_val) 
        val_loss = loss_object(y_true=y_val, y_pred=val_logits)
        losses.append(val_loss)

        #Round off and cast outputs to either  or 1
        val_logits = tf.cast(tf.round(model(x_val)), 'int64')

        #Update validation metrics
        val_acc_metric.update_state(y_val, val_logits)
        val_f1score_metric.update_state(y_val, val_logits)
        
    return losses

# %% Training the Model
epochs = 16
epochs_val_losses, epochs_train_losses = [], []

for epoch in range(epochs):
    print('Start of epoch %d' % (epoch,))
    #Perform Training over all batches of train data
    losses_train = train_data_for_one_epoch(train_dataset, optimizer, loss_object, model, train_acc_metric, train_f1score_metric)

    # Get results from training metrics
    train_acc = train_acc_metric.result()
    train_f1score = train_f1score_metric.result()

    #Perform validation on all batches of test data
    losses_val = perform_validation()

    # Get results from validation metrics
    val_acc = val_acc_metric.result()
    val_f1score = val_f1score_metric.result()

    #Calculate training and validation losses for current epoch
    losses_train_mean = np.mean(losses_train)
    losses_val_mean = np.mean(losses_val)
    epochs_val_losses.append(losses_val_mean)
    epochs_train_losses.append(losses_train_mean)

    print('\n Epcoh %s: Train loss: %.4f  Validation Loss: %.4f, Train Accuracy: %.4f, Validation Accuracy %.4f, Train F1 Score: %.4f, Validation F1 Score: %.4f' % (epoch, float(losses_train_mean), float(losses_val_mean), float(train_acc), float(val_acc), train_f1score, val_f1score))

    #Reset states of all metrics
    train_acc_metric.reset_states()
    val_acc_metric.reset_states()
    val_f1score_metric.reset_states()
    train_f1score_metric.reset_states()


def plot_metrics(train_metric, val_metric, metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0,ylim)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.plot(train_metric,color='blue',label=metric_name)
    plt.plot(val_metric,color='green',label='val_' + metric_name)

plot_metrics(epochs_train_losses, epochs_val_losses, "Loss", "Loss", ylim=1.0)















 
 
 
 
 
 
 