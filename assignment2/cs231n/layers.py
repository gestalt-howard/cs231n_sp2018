from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    
    new_x_shape = (x.shape[0], np.prod(x.shape[1:]))
    out = np.dot(x.reshape(new_x_shape), w) + b

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    new_x_shape = (x.shape[0], np.prod(x.shape[1:]))
    
    dx = np.dot(dout, w.T).reshape(x.shape)
    dw = np.dot(x.reshape(*new_x_shape).T, dout)
    db = np.sum(dout, axis=0)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    
    out = x.clip(min=0)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    
    dx = dout.copy()
    dx[x<=0] = 0
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        
        N, D = x.shape
        # Step 1: calculate mean
        mu = (1/N)*np.sum(x, axis=0)
        # Step 2: subtract mean vector from every training example
        xmu = x-mu
        # Step 3: Square xmu
        sq = np.square(xmu)
        # Step 4: Calculate variance
        var = (1/N)*np.sum(sq, axis=0)
        # Step 5: Add eps for numerical stability, then sqrt
        sqrtvar = np.sqrt(var + eps)
        # Step 6: Invert sqrtvar
        ivar = 1/sqrtvar
        # Step 7: Execute normalization
        xhat = xmu * ivar
        # Step 8: Multiply by gamma
        gammax = gamma * xhat
        # Step 9: Add beta
        out = gammax + beta
        
        running_mean = momentum*running_mean + (1-momentum)*mu
        running_var = momentum*running_var + (1-momentum)*var
        # Store intermediate values
        cache = (x, mu, xhat, gamma, xmu, ivar, sqrtvar, var, eps)
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        
        xhat = (x-running_mean)/np.sqrt(running_var + eps)
        out = gamma*xhat + beta
        
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    
    x, mu, xhat, gamma, xmu, ivar, sqrtvar, var, eps = cache
    N, D = x.shape
    
    # Step 9:
    d_gammax = dout
    dbeta = np.sum(dout, axis=0)
    # Step 8:
    d_xhat = d_gammax*gamma
    dgamma = np.sum(d_gammax*xhat, axis=0)
    # Step 7:
    d_xmu1 = d_xhat*ivar
    d_ivar = np.sum(d_xhat*xmu, axis=0)
    # Step 6:
    d_sqrtvar = d_ivar*(-1/np.square(sqrtvar))
    # Step 5:
    d_var = 0.5*(1/np.sqrt(var+eps))*d_sqrtvar
    # Step 4:
    d_sq = (1/N)*np.ones_like(xhat)*d_var
    # Step 3:
    d_xmu2 = 2*xmu*d_sq
    # Step 2:
    d_x1 = d_xmu1 + d_xmu2
    d_mu = -1*np.sum((d_xmu1+d_xmu2), axis=0)
    # Step 1:
    d_x2 = (1/N)*np.ones_like(xhat)*d_mu
    # Step 0:
    dx = d_x1+d_x2
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    
    x, mu, xhat, gamma, xmu, ivar, sqrtvar, var, eps = cache
    N, D = x.shape
    
    var_adj = var+eps
    
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout*(x-mu)*(var_adj)**(-1/2), axis=0)
    dx = (1/N)*gamma*(var_adj)**(-1/2) * (N*dout - np.sum(dout, axis=0) - (x-mu)*(var_adj)**(-1)*np.sum(dout*(x-mu), axis=0))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    
    N, D = x.shape
    # Step 1: calculate mean
    mu = (1/D)*np.sum(x, axis=1)
    # Step 2: subtract mean vector from every training example
    xmu = x.T-mu
    # Step 3: Square xmu
    sq = np.square(xmu)
    # Step 4: Calculate variance
    var = (1/D)*np.sum(sq, axis=0)
    # Step 5: Add eps for numerical stability, then sqrt
    sqrtvar = np.sqrt(var + eps)
    # Step 6: Invert sqrtvar
    ivar = 1/sqrtvar
    # Step 7: Execute normalization
    xhat = xmu * ivar
    # Step 8: Multiply by gamma
    gammax = gamma * xhat.T
    # Step 9: Add beta
    out = gammax + beta
    
    # Assemble cache
    cache = (x, mu, xhat, gamma, xmu, ivar, sqrtvar, var, eps)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    
    x, mu, xhat, gamma, xmu, ivar, sqrtvar, var, eps = cache
    N, D = x.shape
    
    # Step 9:
    d_gammax = dout
    dbeta = np.sum(dout, axis=0)
    # Step 8:
    d_xhat = d_gammax*gamma
    dgamma = np.sum(d_gammax*xhat.T, axis=0)
    # Step 7:
    d_xmu1 = d_xhat.T*ivar
    d_ivar = np.sum(d_xhat.T*xmu, axis=0)
    # Step 6:
    d_sqrtvar = d_ivar*(-1/np.square(sqrtvar))
    # Step 5:
    d_var = 0.5*(1/np.sqrt(var+eps))*d_sqrtvar
    # Step 4:
    d_sq = (1/D)*np.ones_like(xhat)*d_var
    # Step 3:
    d_xmu2 = 2*xmu*d_sq
    # Step 2:
    d_x1 = d_xmu1 + d_xmu2
    d_mu = -1*np.sum((d_xmu1+d_xmu2), axis=0)
    # Step 1:
    d_x2 = (1/D)*np.ones_like(xhat)*d_mu
    # Step 0:
    dx = (d_x1+d_x2).T
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        
        mask = (np.random.rand(*x.shape)<p)/p
        out = mask*x
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        
        out = x
        
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        
        dx = dout*mask
        
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    
    pad = conv_param.get('pad', 0)
    stride = conv_param.get('stride', 1)
    
    # Pad image with zeros
    padded_x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    
    # Get output dimensions
    out_H = int(1+(H+2*pad-HH)/stride)
    out_W = int(1+(W+2*pad-WW)/stride)
    
    # Perform convolutions
    out = np.zeros((N, F, out_H, out_W))
    
    # NOTE: For loop indexes construct the feature layers
    # NOTE: Feature layers have dimensions (N, F, H', W')
    
    for n in range(N):  # All images
        for f in range(F):  # All filters
            for hh in range(out_H):  # Horizontal positions in final img.
                for ww in range(out_W):  # Vertical positions in final img.
                    
                    # Define indexes for traversing input data
                    h_str = hh*stride
                    h_end = h_str + HH
                    w_str = ww*stride
                    w_end = w_str + WW
                    
                    c_sum = 0
                    for c in range(C):  # Each channel 
                        o_patch = padded_x[n, c, h_str:h_end, w_str:w_end].flatten()
                        f_patch = w[f, c, :, :].flatten()
                        c_sum += np.dot(o_patch, f_patch) 
                    out[n, f, hh, ww] = c_sum + b[f]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache
    
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    N, F, out_H, out_W = dout.shape
    
    pad = conv_param.get('pad', 0)
    stride = conv_param.get('stride', 1)
    
    # Pad image with zeros
    padded_x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    
    # Initialize gradient constructors
    temp_dx = np.zeros_like(padded_x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    # Calculate db
    for i in range(F):
        db[i] = np.sum(dout[:, i, :, :])
    
    # Calculate dw 
    # Iterate through dimensions of feature layers
    for n in range(N):
        for f in range(F):
            for hh in range(out_H):
                for ww in range(out_W):
                    # Define feature layer key value
                    key_val = dout[n, f, hh, ww]
                    
                    # Define indexes for traversing input data
                    h_str = hh*stride
                    h_end = h_str + HH
                    w_str = ww*stride
                    w_end = w_str + WW
                    
                    # Construct dw using running sum
                    for c in range(C):
                        dw[f, c, :, :] += key_val*padded_x[n, c, h_str:h_end, w_str:w_end]
                        
    # Calculate dx
    # Iterate through dimensions of feature layers
    for n in range(N):
        for f in range(F):
            for hh in range(out_H):
                for ww in range(out_W):
                    # Define feature layer key value
                    key_val = dout[n, f, hh, ww]
                    
                    # Define indexes for traversing input data
                    h_str = hh*stride
                    h_end = h_str+HH
                    w_str = ww*stride
                    w_end = w_str+WW
                    
                    # Construct dx using overlapping sums
                    for c in range(C):
                        temp_dx[n, c, h_str:h_end, w_str:w_end] += key_val*w[f, c, :, :]
    
    # Remove padding dimensions from temp_dx 
    dx = temp_dx[:, :, 1:H+1, 1:W+1]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    pool_height = pool_param.get('pool_height', 2)
    pool_width = pool_param.get('pool_width', 2)
    stride = pool_param.get('stride', 2)
    
    N, C, H, W = x.shape
    
    H_out = int(1+(H-pool_height)/stride)
    W_out = int(1+(W-pool_width)/stride)
    
    out = np.zeros((N, C, H_out, W_out))
    
    # Assemble max pooling output
    for n in range(N):
        for c in range(C):
            for hh in range(H_out):
                for ww in range(W_out):
                    # Define indexes for traversing c activation layer
                    h_str = hh*stride
                    h_end = h_str+pool_height
                    w_str = ww*stride
                    w_end = w_str+pool_width
                    
                    out[n, c, hh, ww] = np.max(x[n, c, h_str:h_end, w_str:w_end])
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    
    pool_height = pool_param.get('pool_height', 2)
    pool_width = pool_param.get('pool_width', 2)
    stride = pool_param.get('stride', 2)
    
    N, C, H, W = x.shape
    N, C, H_out, W_out = dout.shape
    
    dx = np.zeros_like(x)
    
    for n in range(N):
        for c in range(C):
            for hh in range(H_out):
                for ww in range(W_out):
                    # Define indexes for traversing c activation layer
                    h_str = hh*stride
                    h_end = h_str+pool_height
                    w_str = ww*stride
                    w_end = w_str+pool_width
                    
                    # Find index of maximum element in pool window
                    flat_idx = np.argmax(x[n, c, h_str:h_end, w_str:w_end])
                    # Reformat index into matrix format
                    max_idx = np.unravel_index(flat_idx, (pool_height, pool_width))
                    # Update gradient for specific window element
                    dx[n, c, h_str:h_end, w_str:w_end][max_idx] = dout[n, c, hh, ww]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    
    N, C, H, W = x.shape
    
    # Restructure x to have channel information in feature slot
    x_t = x.transpose((0, 2, 3, 1))
    x_flat = x_t.reshape((-1, C))
    
    flat_norm, cache = batchnorm_forward(x_flat, gamma, beta, bn_param)
    
    # Reformat flattened array into correct output shape
    out_t = flat_norm.reshape(*x_t.shape)
    out = out_t.transpose(0, 3, 1, 2)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    
    N, C, H, W = dout.shape
    
    dout_t = dout.transpose((0, 2, 3, 1))
    dout_flatten = dout_t.reshape(-1, C)
    
    dx_flatten, dgamma, dbeta = batchnorm_backward(dout_flatten, cache)
    
    dx_t = dx_flatten.reshape(*dout_t.shape)
    dx = dx_t.transpose((0, 3, 1, 2))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer number of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    
    N, C, H, W = x.shape
    ch_per_chunk = int(C/G)
    
    # Format x to isolate G features
    x = x.reshape((N*G, ch_per_chunk*H*W))
    x = x.T
    
    # Step 1: calculate mean
    mu = np.mean(x, axis=0)
    # Step 2: subtract mean vector from every training example
    xmu = x-mu
    # Step 3: Square xmu
    sq = np.square(xmu)
    # Step 4: Calculate variance
    var = np.mean(sq, axis=0)
    # Step 5: Add eps for numerical stability, then sqrt
    sqrtvar = np.sqrt(var + eps)
    # Step 6: Invert sqrtvar
    ivar = 1/sqrtvar
    # Step 7: Execute normalization
    xhat = xmu * ivar
    
    # Reformat x 
    x_norm = xhat.T.reshape((N, C, H, W))
    
    # Step 8: Multiply by gamma
    gammax = gamma * x_norm
    # Step 9: Add beta
    out = gammax + beta
    
    # Assemble cache
    cache = (x, mu, xhat, gamma, xmu, ivar, sqrtvar, var, eps, G)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    
    # All imported matrices in cache are 2D-collapsed and transposed
    x, mu, xhat, gamma, xmu, ivar, sqrtvar, var, eps, G = cache
    N, C, H, W = dout.shape
    
    # Step 9:
    d_gammax = dout
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)  # keepdims retains (1, C, 1, 1) shape
    
    # Step 8:
    d_xhat = d_gammax*gamma
    xhat = xhat.T.reshape((N, C, H, W))
    dgamma = np.sum(d_gammax*xhat, axis=(0, 2, 3), keepdims=True)
    
    # Reformat into 2D and transposed format
    d_xhat = d_xhat.reshape(*x.T.shape).T
    xhat = xhat.reshape(*x.T.shape).T
    N_new, D_new = xhat.shape
    
    # Step 7:
    d_xmu1 = d_xhat*ivar
    d_ivar = np.sum(d_xhat*xmu, axis=0)
    # Step 6:
    d_sqrtvar = d_ivar*(-1/np.square(sqrtvar))
    # Step 5:
    d_var = 0.5*(1/np.sqrt(var+eps))*d_sqrtvar
    # Step 4:
    d_sq = (1/N_new)*np.ones_like(xhat)*d_var
    # Step 3:
    d_xmu2 = 2*xmu*d_sq
    # Step 2:
    d_x1 = d_xmu1 + d_xmu2
    d_mu = -1*np.sum((d_xmu1+d_xmu2), axis=0)
    # Step 1:
    d_x2 = (1/N_new)*np.ones_like(xhat)*d_mu
    # Step 0:
    dx = (d_x1+d_x2).T
    dx = dx.reshape((N, C, H, W))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx