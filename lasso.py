import numpy as np
import random
from matplotlib import pyplot as plt

# diabetes dataset
data = np.loadtxt('crime.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)

# percentages used for training and testing respectively
num_train = int(0.75*n)
num_test = n - num_train

# split data into training set and testing set   
sample_train = data[0:num_train,0:-1]
sample_test = data[num_train:,0:-1]
label_train = data[0:num_train,-1]
label_test = data[num_train:,-1]

# hyper-parameters of your model 
lam = 0.5

def coordinate_descent(sample, label, steps):
    features = np.shape(sample)[1]
    beta = np.random.rand(features)
    count = 0
    output = []
    while count <= max(steps):
        count += 1
        t = random.randint(0, features - 1)
        if t == 0:
            beta_excl = beta.copy()
            beta_excl[0] = 0
            next_beta = 0
            for i in range(1, features):
                next_beta += np.matmul(np.transpose(sample[i]), beta_excl) - label[i]
            beta[0] = -1 * next_beta / features
        else:
            beta_excl = beta.copy()
            beta_excl[t] = 0
            xby = np.matmul(sample, beta_excl) - label
            delta = np.matmul(-2 * np.transpose(sample[:, t]), xby)
            if delta > lam:
                beta[t] = (delta - lam) / (2 * np.matmul(np.transpose(sample[:, t]), sample[:, t]))
            elif delta < -1 * lam:
                beta[t] = (delta + lam) / (2 * np.matmul(np.transpose(sample[:, t]), sample[:, t]))
            else:
                beta[t] = 0
        
        if count in steps:
            model = predict(sample_test, beta)
            output.append(root_mean_squared_error(label_test, model))
    
    return beta, output

def sparsity(beta, features):
    return (features - np.count_nonzero(beta)) / features

def predict(sample, beta):
    return np.matmul(sample[:, np.newaxis], beta)

def root_mean_squared_error(true, pred):
    size = np.size(true)
    if size == 0:
        return 0
    sum = 0
    for i in range(size):
        sum += pow(true[i] - pred[i], 2)
    return pow(sum / size, 0.5)[0]

steps = [i * 10000 for i in range(1, 21)]
beta, output = coordinate_descent(sample_train, label_train, steps)

fig, ax = plt.subplots()

ax.plot(steps, output)
plt.xlabel('CD updates')
plt.ylabel('Testing error')
plt.show()