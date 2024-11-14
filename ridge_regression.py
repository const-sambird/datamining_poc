import numpy as np

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
alpha = 0.5

def rr(sample, label, weights, l):
    xt = np.transpose(sample)
    xtx = np.matmul(xt, sample)
    lnw = l * weights
    weights[0][0] = 0
    xty = np.matmul(xt, label)
    return np.matmul(np.linalg.inv(xtx + lnw), xty)

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

def k_fold_cross_validation(k: int, sample: list[list[float]], label: list[float], lambdas: list[float]):
    """
    Divides the sample set into k folds.

    Takes:
    * `k`: the number of folds to make
    * `sample`: the training set
    * `label`: the training labels
    * `lambdas`: an array of Numbers of values of lambda to test

    Returns the validation errors associated with each lambda
    """
    samples = list()
    labels = list()
    for i in range(k):
        samples.append(sample[i::k,0:-1])
        labels.append(label[i::k])
    
    errors = list()

    for l in lambdas:
        current_errors = []
        for i in range(k):
            sample_test = samples[i]
            label_test = labels[i]

            for j in range(k):
                if i == j: continue # this is the test set. don't train it
                # set the weights to 1
                weights = np.identity(np.shape(samples[j])[1])
                model = rr(samples[j], labels[j], weights, l)
                prediction = predict(sample_test, model)
                current_errors.append(root_mean_squared_error(label_test, prediction))
        
        errors.append((l, np.average(current_errors), np.std(current_errors)))
    
    return errors

l = k_fold_cross_validation(10, sample_train, label_train, [0, 0.25, 0.5, 0.75, 1])

for value in l:
    lda, mean, stddev = value
    print('λ = %.2f: %.4f ± %.4f' % (lda, mean, stddev))
