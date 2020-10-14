import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")

data.info()

"""
Data columns (total 33 columns):
 #   Column                   Non-Null Count  Dtype  
---  ------                   --------------  -----  
 0   id                       569 non-null    int64  
.
.
.
 32  Unnamed: 32              0 non-null      float64
"""
data.drop(["Unnamed: 32", "id"], axis = 1, inplace = True)
# data.head(10)
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis = 1)

# %% Normalization
x_normalized = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

x_data.head()
"""
x_data.head()
Out[9]: 
   radius_mean  texture_mean  ...  symmetry_worst  fractal_dimension_worst
0        17.99         10.38  ...          0.4601                  0.11890
1        20.57         17.77  ...          0.2750                  0.08902
2        19.69         21.25  ...          0.3613                  0.08758
3        11.42         20.38  ...          0.6638                  0.17300
4        20.29         14.34  ...          0.2364                  0.07678
"""
x_normalized.head()
"""
x_normalized.head()
Out[10]: 
   radius_mean  texture_mean  ...  symmetry_worst  fractal_dimension_worst
0     0.521037      0.022658  ...        0.598462                 0.418864
1     0.643144      0.272574  ...        0.233590                 0.222878
2     0.601496      0.390260  ...        0.403706                 0.213433
3     0.210090      0.360839  ...        1.000000                 0.773711
4     0.629893      0.156578  ...        0.157500                 0.142595
"""
# %% train test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_normalized,y,test_size = 0.25, random_state = 42)
# test size & random state can be changed, test size can be choosen as 0.2 or 0.18
# sklearn randomly splits, with given state data will be splitted with same random pattern.
# rows as features
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

# %% Parameter Initialize
"""
If all the weights were initialized to zero, 
backpropagation will not work as expected because the gradient for the intermediate neurons
and starting neurons will die out(become zero) and will not update ever.
"""
def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1), 0.01) # init 0.01
    b = np.zeros(1)
    return w,b

def sigmoid(n):
    y_hat = 1 / (1 + np.exp(-n))
    return y_hat

# %% 
def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    z = np.dot(w.T,x_train) + b
    y_hat = sigmoid(z)
    loss = -(y_train*np.log(y_hat)+(1-y_train)*np.log(1-y_hat))
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling
    # Once cost is calculated, forward prop. is completed.
    
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_hat-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_hat-y_train)/x_train.shape[1] # x_train.shape[1] is for scaling
    # x_train.shape[1] = 426
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
    return cost,gradients

# Updating(learning) parameters
def update(w, b, x_train, y_train, learning_rate,number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iteration):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 13 == 0: # that's arbitrary, you can set it differently
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
    # we update(learn) parameters weights and bias
    parameters = {"weight": w,"bias": b}
    
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()
    return parameters, gradients, cost_list

# prediction
def predict(w,b,x_test):
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.64, our prediction is one - true (y_hat=1),
    # if z is smaller than 0.64, our prediction is sign zero - false (y_hat=0),
    for i in range(z.shape[1]):
        if z[0,i]<= 0.64:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction

#implementing logistic regression
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate , num_iterations):
    # initialize
    dimension =  x_train.shape[0]
    w,b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    # Print accuracy
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 10, num_iterations = 250)
#learning rate & # of iterations are arbitrary, hyperparameter tuning.
"""
Cost after iteration 0: 0.693035
Cost after iteration 13: 0.219091
Cost after iteration 26: 0.134309
Cost after iteration 39: 0.120401
Cost after iteration 52: 0.111365
Cost after iteration 65: 0.105184
Cost after iteration 78: 0.100691
Cost after iteration 91: 0.097229
Cost after iteration 104: 0.094427
Cost after iteration 117: 0.092072
Cost after iteration 130: 0.090038
Cost after iteration 143: 0.088245
Cost after iteration 156: 0.086640
Cost after iteration 169: 0.085189
Cost after iteration 182: 0.083863
Cost after iteration 195: 0.082645
Cost after iteration 208: 0.081519
Cost after iteration 221: 0.080472
Cost after iteration 234: 0.079495
Cost after iteration 247: 0.078580
test accuracy: 98.6013986013986 %
"""
# %%
