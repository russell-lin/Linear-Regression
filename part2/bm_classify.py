import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0

    b = 0
    if b0 is not None:
        b = b0

    y_list = y.tolist()
    for i in range(len(y_list)):
        if y_list[i] == 0:
            y_list[i] = -1
    new_y = np.array(y_list)


    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        for i in range(max_iterations):
            error = new_y * (w.dot(X.T) + b)
            error = np.where(error <= 0, 1 , 0)
            w_gradient = (error * new_y).dot(X)
            b_gradient = np.sum(error * new_y)
            w += step_size * w_gradient / N
            b += step_size * b_gradient / N


        ############################################

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        for i in range(max_iterations):
            error = - new_y * (w.dot(X.T) + b)
            w_gradient = -(sigmoid(error) * new_y).dot(X)
            b_gradient = np.sum(-sigmoid(error) * new_y)
            w -= step_size * w_gradient / N
            b -= step_size * b_gradient / N
        ############################################


    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b

def sigmoid(z):

    """
    Inputs:
    - z: a numpy array or a float number

    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value = z
    value = 1 / (1 + np.exp(-z))

    ############################################

    return value

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic

    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape

    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        preds = w.dot(X.T) + b
        preds = np.array([1 if i >= 0 else 0 for i in preds])
        ############################################


    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        preds = sigmoid(w.dot(X.T) + b)
        preds = np.array([1 if i >= 0.5 else 0 for i in preds])
        ############################################


    else:
        raise "Loss Function is undefined."


    assert preds.shape == (N,)
    return preds

def softmax(x):
    x = np.exp(x - np.amax(x))
    denom = np.sum(x, axis=1)
    return (x.T / denom).T

def multiclass_train(X, y, C,
                     w0=None,
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5,
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        for i in range(max_iterations):
            index = np.random.choice(N)
            error = w.dot(X[index]) + b
            error -= np.max(error)
            soft_max = np.exp(error)
            soft_max = soft_max / np.sum(soft_max)
            soft_max[y[index]] -= 1
            w_gradient = soft_max.reshape(C,1).dot(X[index].reshape(1,D))
            b_gradient = soft_max
            w -= step_size * w_gradient
            b -= step_size * b_gradient
        ############################################



    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        y = np.eye(C)[y]
        for i in range(max_iterations):
            P = softmax((w.dot(X.T)).T + b) - y
            w_gradient = P.T.dot(X)
            b_gradient = np.sum(P, axis=0)
            w -= step_size * w_gradient / N
            b -= step_size * b_gradient / N

        ############################################


    else:
        raise "Type of Gradient Descent is undefined."


    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D
    - b: bias terms of the trained multinomial classifier, length of C

    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    preds = np.zeros(N)
    preds = softmax((w.dot(X.T)).T + b)
    preds = np.argmax(preds, axis=1)
    ############################################

    assert preds.shape == (N,)
    return preds




