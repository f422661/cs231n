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
  n_sample = X.shape[0]
  n_class = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  for i in range(n_sample):

    pred = np.dot(X[i,:],W)
 
    # shift the logit
    pred -= np.max(pred)
    p = np.exp(pred)/np.sum(np.exp(pred))
    loss += -np.log(p[y[i]])

    # The gradient of dL over dF
    p[y[i]] -= 1 

    for j in range(n_class):
      dW[:,j] += p[j]*X[i,:]



  loss = loss/n_sample
  dW = dW/n_sample

  
  loss += reg * np.sum(W * W)
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  n_sample = X.shape[0]
  n_class = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  ## The input is all batch
  ## Tip: using np.tile
  pred = np.dot(X,W) #(N,C)
  shift = np.max(pred,axis = 1)
  pred -= np.tile(shift,(n_class,1)).T
  ## normalize
  p = np.exp(pred)/np.tile(np.sum(np.exp(pred),axis = 1),(n_class,1)).T #(N,C)
  loss = np.average(-np.log(p[np.arange(p.shape[0]),y]))

  ## The gradient of dL over dF
  p[np.arange(p.shape[0]),y]-=1
  dW = np.dot(X.T,p)
  dW = dW/n_sample



  # regularization
  loss += reg * np.sum(W * W)
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

