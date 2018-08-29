import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        loss_counter = 0
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                loss_counter += 1
                dW[:, j] += X[i]  # Gradient update for incorrect class
        # Calculate gradient of loss wrt correct class
        dW[:, y[i]] += (-loss_counter*X[i]).T

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * 2*W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    
    # Calculate loss
    scores = np.dot(X, W)
    correct_scores = scores[np.arange(scores.shape[0]), y]
    margins = np.maximum(0, scores - np.matrix(correct_scores).T + 1)
    margins[np.arange(scores.shape[0]), y] = 0
    loss = np.mean(np.sum(margins, axis=1))
    # Add regularization
    loss += reg*np.sum(W*W)
    
    
    # Calculate gradient
    binary_mask = margins
    binary_mask[margins>0] = 1
    nonzero_count = np.sum(binary_mask, axis=1)
    binary_mask[np.arange(scores.shape[0]), y] = -np.matrix(nonzero_count).T
    dW = np.dot(X.T, binary_mask)
    dW /= X.shape[0]
    # Add regularization
    dW += reg*2*W

    return loss, dW
