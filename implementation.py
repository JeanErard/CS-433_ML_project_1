import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma

def load_data(path, header=False) :
    path_data = path
    data = np.genfromtxt(path_data, delimiter=',', skip_header=1)
    output = np.delete(data,  [0], axis=1)
    features = None
    if header :
        features = np.genfromtxt(path_data, dtype=str, delimiter=',', max_rows=1)
        features = np.delete(features, [0])
    return output.astype(np.float32), features

def plot_features(data, features) :
    size = len(features)
    num_col = 6
    num_row = int(np.ceil(size/num_col))
    f, axs = plt.subplots(num_row, num_col, figsize=(20, 150))
    for i, feature in enumerate(features) :
        value = data[:,i]
        value_filt = value[~np.isnan(value)]
        ax = axs[i//num_col][i%num_col]
        ax.hist(value_filt, bins=10, color='b')
        """bins = int(np.sqrt(len(value_filt)))"""
        ax.set_xlabel(f'{feature}')
        ax.set_ylabel('Density')
        ax.set_title(f"{feature} ({i})")
    
    plt.tight_layout()
    plt.show()
        
def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.

    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.

    >>> split_data(np.arange(13), np.arange(13), 0.8, 1)
    (array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]), array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]))
    """
    # set seed
    np.random.seed(seed)

    nbre_data = int(np.floor(ratio*len(y)))
    indices = np.random.permutation(len(y))
    ind_tr = indices[:nbre_data]
    ind_te = indices[nbre_data:]
    x_tr = x[ind_tr]
    y_tr = y[ind_tr]
    x_te = x[ind_te]
    y_te = y[ind_te]
    
    return x_tr, x_te, y_tr, y_te

def normalize(x) :
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    norm_x = (x-mean) / std
    return norm_x, mean, std


def remove_col_full_nan(x, header, threshold = 1) :
    find_nan = np.isnan(x)
    count_nan = np.count_nonzero(find_nan, axis = 0)
    col_del = np.where((count_nan / x.shape[0]) >= threshold)
    out_x = np.delete(x,col_del, axis=1)
    out_header = np.delete(header, col_del)
    return out_x, out_header, col_del
    
def const_column(x, header, threshold=1) :
    col_to_delete = []
    for col in range(x.shape[1]) :
        column = x[:,col]
        mask = ~np.isnan(column)
        counts = np.unique(column[mask], return_counts=True)[1]
        if (len(counts) == 1) or (np.max(counts)/np.sum(counts) >= threshold) :
                col_to_delete.append(col)
    out_x = np.delete(x, col_to_delete, axis =1)
    out_header = np.delete(header, col_to_delete)
    return out_x, out_header, col_to_delete

"""def correlation(x, header, threshold=1) :
    corr = ma.corrcoef(ma.masked_invalid(x), rowvar=False)
    corr = ma.asarray(corr).copy()
    corr = np.triu(corr, k=1)
    print(corr)
    corr_feature = np.where(ma.filled(corr >= threshold, False))
    print(corr_feature)"""
    
def pairwise_corr(x, header, threshold=1):
    n_features = x.shape[1]
    col_rem = []
    
    for i in range(n_features):
        if i in col_rem :
            continue
        for j in range(i+1, n_features):
            xi, xj = x[:, i], x[:, j]
            mask = ~np.isnan(xi) & ~np.isnan(xj)
            if mask.sum() > 1:
                r = np.corrcoef(xi[mask], xj[mask])[0, 1]
                if np.abs(r) >= threshold :
                    col_rem.append(j)
    
    out_x = np.delete(x, col_rem, axis =1)
    out_header = np.delete(header, col_rem)
    return out_x, out_header, col_rem

def replace_nan(x, type='zero') :
    out = np.zeros(np.shape(x))
    if type == 'zero' :
        out = np.where(np.isnan(x), 0, x)
    if type == 'mean' :
        for i in range(x.shape[1]) :
            col = x[:,i].copy()
            mean = np.nanmean(col)
            out[:,i] = np.where(np.isnan(col), mean, col)
    return out

def compute_loss_mse(y, x, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    y_predicted = x.dot(w)
    error = y - y_predicted
    mse = ((1 / (2 * len(y))) * error.T.dot(error))
    return mse

def least_squares(y, x):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    gram_matrix = x.T @ x
    sol = x.T @ y
    w = np.linalg.solve(gram_matrix, sol)
    mse = compute_loss_mse(y, x, w)
    
    return w, mse[0]

def ridge_regression(y, x, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    w = np.linalg.solve(x.T @ x + 2*len(y)*lambda_*np.eye(np.shape(x)[1]), x.T @ y)
    mse = compute_loss_mse(y, x, w)
    return w, mse[0]

def mean_squared_error_gd(y, x, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters + 1 containing the model parameters as numpy arrays of shape (2, ),
            for each iteration of GD (as well as the final weights)
    """
    w = initial_w
    for n_iter in range(max_iters):
        #Compute the gradient
        error = y - x.dot(w)
        gradient_loss = ((-1) / len(y)) * x.T.dot(error)
        #update w by gradient
        w = w - gamma*gradient_loss

    mse = compute_loss_mse(y, x, w)
    return w, mse

def mean_squared_error_sgd(y, x, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """
    w = initial_w

    for n_iter in range(max_iters):
        for i in range(y.shape[0]):
            point_x = x[i,:]
            point_y = y[i]
            gradient_loss = ((-1) / len(y)) * point_x.T.dot(point_y - point_x.dot(w)) 

            #update w
            w = w - gamma*gradient_loss
        
    mse = compute_loss_mse(y, x, w)
    return w, mse

def lasso_regression(y, x, lambda_) :
    x