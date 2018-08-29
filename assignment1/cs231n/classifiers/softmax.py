import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    for i in range(num_train):
        all_scores = np.zeros(num_classes)
        for j in range(num_classes):
            all_scores[j] = np.exp(np.dot(X[i], W[:, j]))
            
        # Update loss and gradient
        scaled_scores = all_scores/np.sum(all_scores)
        for idx, ss in enumerate(scaled_scores):
            if idx == y[i]:
                dW[:, idx] += (ss-1)*X[i]
                loss += -1.0*np.log(ss)
            else:
                dW[:, idx] += ss*X[i]
                
    # Normalize
    loss /= num_train
    dW /= num_train
    # Add regularization
    loss += reg * np.sum(W*W)
    dW += reg * 2*W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_class = W.shape[1]
    
    scores = np.exp(np.dot(X, W))
    score_sums = np.sum(scores, axis=1)
    
    # Calculate loss
    loss = np.sum(-1.0*(np.log(scores[np.arange(num_train), y] / score_sums)))
    # Calculate gradient
    dW_weights = scores / score_sums[:, None]
    dW_weights[np.arange(num_train), y] = dW_weights[np.arange(num_train), y]-1
    dW = np.dot(X.T, dW_weights)
    
    # Normalize
    loss /= num_train
    dW /= num_train
    # Add regularization
    loss += reg * np.sum(W*W)
    dW += reg* 2*W

    return loss, dW

