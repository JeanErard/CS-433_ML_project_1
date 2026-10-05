import csv
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy.ma as ma

def load_data(path, header=False, intercept=False) :
    path_data = path
    data = np.genfromtxt(path_data, delimiter=',', skip_header=1)
    output = np.delete(data,  [0], axis=1)
    features = None
    if header :
        features = np.genfromtxt(path_data, dtype=str, delimiter=',', max_rows=1)
        features = np.delete(features, [0])
    final_feat = None
    if intercept :
        output = np.c_[np.ones(output.shape[0]), output]
        final_feat = np.insert(features, 0, 'intercept')
    return output, final_feat

def plot_features(data, features) :
    size = len(features)
    num_col = 6
    num_row = int(np.ceil(size/num_col))
    f, axs = plt.subplots(num_row, num_col, figsize=(20, 150))
    for i, feature in enumerate(features) :
        value = data[:,i]
        value_filt = value[~np.isnan(value)]
        ax = axs[i//num_col][i%num_col]
        ax.hist(value_filt, bins=int(np.sqrt(len(value_filt))), color='b')
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

## Cross-validation
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    
    return np.array(k_indices)

def cross_validation_one_iteration(y, x, k_indices, k, lambda_):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    >>> cross_validation(np.array([1.,2.,3.,4.]), np.array([6.,7.,8.,9.]), np.array([[3,2], [0,1]]), 1, 2, 3)
    (0.019866645527596753, 0.33555914361295197)
    """
    test_index = k_indices[k]
    x_test = x[test_index]
    y_test = y[test_index]
    train_index =  np.delete(k_indices, k, axis=0).ravel()
    x_train = x[train_index]
    y_train = y[train_index]

    w, loss = ridge_regression(y_train, x_train, lambda_)
    
    loss_tr = np.sqrt(2*loss)
    loss_te = np.sqrt(2*compute_loss_mse(y_test, x_test, w))
    return loss_tr, loss_te

def cross_validation(y, x, k_fold, lambdas):
    """cross validation over regularisation parameter lambda.

    Args:
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """

    seed = 12
    k_fold = k_fold
    lambdas = lambdas
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []

    for lambda_ in lambdas :
        loss_lambda_tr = 0
        loss_lambda_te = 0
        
        for k in range(k_fold) :
            loss_tr, loss_te = cross_validation_one_iteration(y, x, k_indices, k, lambda_)
            loss_lambda_tr += loss_tr
            loss_lambda_te += loss_te
            
        rmse_tr.append(loss_lambda_tr/k_fold)
        rmse_te.append(loss_lambda_te/k_fold)
    
    best_rmse = np.min(rmse_te)
    index = np.where(rmse_te == best_rmse)
    if len(index) > 1 :
        index = index[0]
    best_lambda = lambdas[index]
    plot_lambda(np.squeeze(rmse_tr), np.squeeze(rmse_te), lambdas)

    return best_lambda, best_rmse

def plot_lambda(loss_tr, loss_te, param) :
    plt.semilogx(param, loss_tr, marker=".", color="b", label="train error")
    plt.semilogx(param, loss_te, marker=".", color="r", label="test error")
    plt.xlabel("lambda")
    plt.ylabel("r mse")
    # plt.xlim(1e-4, 1)
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
    
## classification

def compute_logistic_loss(y, x, w):
    y = y.squeeze()
    y_pred = x @ w
    log_loss = np.mean(np.logaddexp(0, -y * y_pred)) #log(1 + exp(-y y_pred))
    return log_loss

def gd_logistic_loss_pen(y, x, initial_w, max_iters, gamma, lambda_) :
    y = y.squeeze()
    w = initial_w.copy()
    
    for n_iter in range(max_iters):
        for i in range(y.shape[0]):
            point_x, point_y = x[i,:], y[i]
            # grad = -y * x * sigma(-y x.w)
            gradient_loss = -point_y * point_x * sigma_function(-point_y * (point_x@w)) + 2 * lambda_ * w
    
            #update w
            w = w - gamma*gradient_loss
            
    return w, compute_logistic_loss(y, x, w)
    
def accuracy(y, y_pred):
    return np.mean(y == y_pred)

def threshold_prediction(y_pred, threshold=0) :
    return np.where(y_pred >= threshold, 1, -1)

def cross_validation_log(y, x, k_fold, lambdas, gammas):
    """cross validation over regularisation parameter lambda.

    Args:
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """

    seed = 12
    k_fold = k_fold
    lambdas = lambdas
    gammas = gammas
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = np.zeros((len(lambdas),len(gammas)))
    rmse_te = np.zeros((len(lambdas),len(gammas)))
    best_loss = []
    best_gamma = []

    for i, lambda_ in enumerate(lambdas) :
        rmse_lam_te =[]
        for j, gamma in enumerate(gammas) :
            rmse_tr_k = []
            rmse_te_k = []
            for k in range(k_fold) :
                loss_tr, loss_te = cross_validation_one_iteration_log(y, x, k_indices, k, lambda_, gamma)
                rmse_tr_k.append(loss_tr)
                rmse_te_k.append(loss_te)
            rmse_tr[i,j] = np.mean(rmse_tr_k)
            rmse_te[i,j] = np.mean(rmse_te_k)
            rmse_lam_te.append(np.mean(rmse_te_k))
        best_gamma.append(gammas[np.argmin(rmse_lam_te)])
        best_loss.append(np.min(rmse_lam_te))
    best_rmse = np.min(best_loss)
    index = np.argmin(best_loss)
    best_lambda = lambdas[index]
    best_gamma = best_gamma[index]
    plot_lambda(np.squeeze(rmse_tr[index,:]), np.squeeze(rmse_te[index,:]), gammas)
    plot_lambda(np.squeeze(rmse_tr[:,np.where(best_gamma == gammas)]), np.squeeze(rmse_te[:, np.where(best_gamma == gammas)]), lambdas)

    return best_lambda, best_gamma, best_rmse

def cross_validation_one_iteration_log(y, x, k_indices, k, lambda_, gamma):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    >>> cross_validation(np.array([1.,2.,3.,4.]), np.array([6.,7.,8.,9.]), np.array([[3,2], [0,1]]), 1, 2, 3)
    (0.019866645527596753, 0.33555914361295197)
    """
    test_index = k_indices[k]
    x_test = x[test_index]
    y_test = y[test_index]
    train_index =  np.delete(k_indices, k, axis=0).ravel()
    x_train = x[train_index]
    y_train = y[train_index]

    w, loss = gd_logistic_loss_pen(y_train, x_train,  np.zeros(x_train.shape[1]), 30, gamma, lambda_)
    
    loss_tr = np.sqrt(2*loss)
    loss_te = np.sqrt(2*compute_logistic_loss(y_test, x_test, w))
    return loss_tr, loss_te

def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})
            
def sigma_function(t) :
    return 1/(1 + np.exp(-t))

def hessian_log(y, x, w, lambda_=0) :
    N, D = x.shape
    s = sigma_function(x@w)
    p = s * (1 - s)
    H = (x.T @ (x * p[:,None])) / N + 2 * lambda_ * np.eye(D) #p[:,None] transform shape (N,) to (N,1) equivalent to np.newaxis
    return H

def gradient_log(y, x, w, lambda_=0) :
    N, D = x.shape
    r = y * sigma_function(-y * (x@w))
    grad = -(x.T @ r) / N + 2*lambda_*w
    return grad

def newton_log_pen(y, x, initial_w, max_iters, lambda_) :
    y = y.squeeze()
    N, D = x.shape
    w = initial_w.copy()
    
    for _ in range(max_iters) :
        grad = gradient_log(y, x, w, lambda_)
        H = hessian_log(y, x, w, lambda_)
        w = w - np.linalg.solve(H, grad)
        
    return w, compute_logistic_loss(y, x, w)

def cross_val_new(y, x, k_fold, lambdas) :
    seed = 12
    k_fold = k_fold
    lambdas = lambdas
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    log_loss_te = []
    
    for lambda_ in lambdas :
        loss_k = []
        for k in range(k_fold) :
            _, loss_te = cross_val_one_log(y, x, k_indices, k, lambda_)
            loss_k.append(loss_te)
        log_loss_te.append(np.mean(loss_k))
    
    best_loss = np.min(log_loss_te)
    index = np.argmin(log_loss_te)
    best_lambda = lambdas[index]
    plot_lambda(np.zeros(len(lambdas)),log_loss_te, lambdas)
    
    return best_lambda, best_loss

def cross_val_one_log(y, x, k_indices, k, lambda_) :
    test_index = k_indices[k]
    x_test = x[test_index]
    y_test = y[test_index]
    train_index =  np.delete(k_indices, k, axis=0).ravel()
    x_train = x[train_index]
    y_train = y[train_index]

    w, _ = newton_log_pen(y_train, x_train,  np.zeros(x_train.shape[1]), 10, lambda_)
    
    loss_te = compute_logistic_loss(y_test, x_test, w)
    return _, loss_te
