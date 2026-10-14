import csv
import numpy as np
import os
import matplotlib.pyplot as plt

def load_data(path, header=False) :
    """
    Load data from a csv file into a numpy array and drop the first column (the ids).
    The first row is always skipped as a header. Optionally returns the column names.
 
    Args:
        path (str): path of the csv file.
        header (bool): if True, the feature names are also read. Default is False.
 
    Returns:
        output (np.ndarray): float array of shape (N, D-1), the first column removed.
        features (np.ndarray or None): array of str of shape (D-1,) with the feature names, or None if header is False.
    """
    path_data = path
    data = np.genfromtxt(path_data, delimiter=',', skip_header=1)
    output = np.delete(data,  [0], axis=1)
    features = None
    if header :
        features = np.genfromtxt(path_data, dtype=str, delimiter=',', max_rows=1)
        features = np.delete(features, [0])
    return output, features

def get_ind_for_sub(path) :
    """
    Read a csv file and return only its first column, i.e. the sample ids needed to build a submission file.
 
    Args:
        path (str): path of the csv file.
 
    Returns:
        np.ndarray: array of shape (N,) containing the ids.
    """
    path_data = path
    data = np.genfromtxt(path_data, delimiter=',', skip_header=1)
    return data[:,0]

def add_intercept(x, header = None):
    """
    Add a column of ones in front of the data matrix (bias term) and, optionally, prepend 'intercept' to the feature names.
 
    Args:
        x (np.ndarray): data matrix of shape (N, D).
        header (np.ndarray of str or None): feature names of shape (D,).
 
    Returns:
        tx (np.ndarray): data matrix of shape (N, D+1).
        header (np.ndarray of str or None): updated feature names, or None.
    """
    tx = np.c_[np.ones(x.shape[0]), x]
    if header is not None :
        header = np.insert(header, 0, 'intercept')
    return tx, header

def plot_features(data, features) :
    """
    Plot one histogram per feature in a grid of 6 columns. NaN values are ignored and the number of bins is the square root of the number of valid values.
 
    Args:
        data (np.ndarray): data matrix of shape (N, D).
        features (sequence of str): feature names, length D.
 
    Returns:
        None: displays a matplotlib figure.
    """
    size = len(features)
    num_col = 6
    num_row = int(np.ceil(size/num_col))
    f, axs = plt.subplots(num_row, num_col, figsize=(20, 150))
    for i, feature in enumerate(features) :
        value = data[:,i]
        value_filt = value[~np.isnan(value)]
        ax = axs[i//num_col][i%num_col]
        ax.hist(value_filt, bins=int(np.sqrt(len(value_filt))), color='b')
        ax.set_xlabel(f'{feature}')
        ax.set_ylabel('Density')
        ax.set_title(f"{feature} ({i})")
    
    plt.tight_layout()
    plt.show()
        
def split_data(x, y, ratio, seed=1):
    """
    Randomly split a dataset into a training set and a test set. 
    A fraction 'ratio' of the samples (rounded down) goes to the training set.
 
    Args:
        x (np.ndarray): data, first dimension is N (number of samples).
        y (np.ndarray): labels of shape (N,).
        ratio (float): fraction of samples used for training, in [0, 1].
        seed (int): random seed. Default is 1.
 
    Returns:
        x_tr (np.ndarray): training data.
        x_te (np.ndarray): test data.
        y_tr (np.ndarray): training labels.
        y_te (np.ndarray): test labels.
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

def normalize(x, index_cont = None) :
    """
    Standardize the data to mean 0 and standard deviation 1, ignoring NaN values.
    If 'index_cont' is given, only these columns are standardized and the others are left unchanged.
 
    Args:
        x (np.ndarray): data matrix of shape (N, D).
        index_cont (list or np.ndarray of int, or None): indices of the columns to standardize. 
            If None, all columns are standardized.
 
    Returns:
        norm_x (np.ndarray): standardized matrix of shape (N, D).
        mean (np.ndarray): mean of each standardized column.
        std (np.ndarray): standard deviation of each standardized column
            (mean and std can be reused on test data).
    """
    x_int = x.copy() if index_cont is None else x[:,index_cont]
    mean = np.nanmean(x_int, axis=0)
    std = np.nanstd(x_int, axis=0)
    if index_cont is None :
        norm_x = (x_int-mean) / std 
    else :
        norm_x = x.copy()
        norm_x[:, index_cont] = (x_int - mean) / std
    return norm_x, mean, std


def remove_col_full_nan(x, header, threshold = 1) :
    """
    Remove the columns whose proportion of NaN values is greater than or equal to 'threshold'.
 
    Args:
        x (np.ndarray): data matrix of shape (N, D).
        header (np.ndarray of str): feature names of shape (D,).
        threshold (float): proportion of NaN in [0, 1] above which a column is removed. 
            Default is 1.
 
    Returns:
        out_x (np.ndarray): matrix without the removed columns.
        out_header (np.ndarray of str): feature names without the removed columns.
        col_del (tuple of np.ndarray): indices of the removed columns (output of np.where).
    """
    find_nan = np.isnan(x)
    count_nan = np.count_nonzero(find_nan, axis = 0)
    col_del = np.where((count_nan / x.shape[0]) >= threshold)
    out_x = np.delete(x,col_del, axis=1)
    out_header = np.delete(header, col_del)
    return out_x, out_header, col_del
    
def const_column(x, header, threshold=1) :
    """
    Remove the columns that are (almost) constant, i.e. those for which the most frequent non-NaN value represents at least 'threshold' of the non-NaN values.
 
    Args:
        x (np.ndarray): data matrix of shape (N, D).
        header (np.ndarray of str): feature names of shape (D,).
        threshold (float): proportion in [0, 1]. Default is 1.
 
    Returns:
        out_x (np.ndarray): matrix without the removed columns.
        out_header (np.ndarray of str): feature names without the removed columns.
        col_to_delete (list of int): indices of the removed columns.
    """
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
    
def pairwise_corr(x, header, threshold=1):
    """
    Remove the columns that are highly correlated with an earlier column.
    The correlation is computed pair by pair, only on the rows where both columns are non-NaN, so missing values are tolerated. 
    A column is removed if the absolute correlation is greater than or equal to 'threshold'.
 
    Args:
        x (np.ndarray): data matrix of shape (N, D).
        header (np.ndarray of str): feature names of shape (D,).
        threshold (float): absolute correlation threshold. Default is 1.
 
    Returns:
        out_x (np.ndarray): matrix without the removed columns.
        out_header (np.ndarray of str): feature names without the removed columns.
        col_rem (list of int): indices of the removed columns.
    """
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

def del_correlate(x, header, threshold = 1) :
    """
    Remove the highly correlated columns, like `pairwise_corr`, but compute the
    whole correlation matrix at once with np.corrcoef. 
    It is faster but does not handle NaN values (impute them first).
 
    Args:
        x (np.ndarray): data matrix of shape (N, D) without NaN.
        header (np.ndarray of str): feature names of shape (D,).
        threshold (float): absolute correlation threshold. Default is 1.
 
    Returns:
        out_x (np.ndarray): matrix without the removed columns.
        out_header (np.ndarray of str): feature names without the removed columns.
        del_ind (list of int): indices of the removed columns.
    """
    data = x.astype(float)
    corr = np.abs(np.corrcoef(data.T))
    del_ind = []
    
    for i in range(x.shape[1]) :
        if i in del_ind :
            continue
        for j in range(i+1, x.shape[1]) :
            if corr[i, j] >= threshold :
                del_ind.append(j)
    return np.delete(x, del_ind, axis=1), np.delete(header, del_ind), del_ind

def replace_nan(x, type='zero') :
    """
    Replace the NaN values of a matrix by 0 or by the mean of their column.
 
    Args:
        x (np.ndarray): data matrix of shape (N, D).
        type (str): 'zero' to replace by 0, 'mean' to replace by the column mean (ignoring NaN). 
            Any other value returns a matrix of zeros.
 
    Returns:
        np.ndarray: matrix of shape (N, D) without NaN.
    """
    out = np.zeros(np.shape(x))
    if type == 'zero' :
        out = np.where(np.isnan(x), 0, x)
    if type == 'mean' :
        for i in range(x.shape[1]) :
            col = x[:,i].copy()
            mean = np.nanmean(col)
            out[:,i] = np.where(np.isnan(col), mean, col)
    return out

def one_hot_encoder(x, header, numb_cat = 9, index_for_continous = False) :
    """
    One-hot encode every non-binary column that has at most `numb_cat` distinct values (NaN counts as a category and gets its own '_nan' indicator column).
    The other columns are kept as they are.
 
    Args:
        x (np.ndarray): data matrix of shape (N, D).
        header (sequence of str): feature names, length D.
        numb_cat (int): maximum number of categories for a column to be encoded.
            Default is 9.
        index_for_continous (bool): if True, also return the indices of the columns
            that were not encoded. Default is False.
 
    Returns:
        new_data (np.ndarray): encoded data matrix.
        new_header (list of str): names of the new columns.
        cont_index (list of int or None): indices of the non-encoded columns in new_data, or None if index_for_continous is False.
        oh_metadata (dict): for each encoded column index, a dict with the keys 'categories' (np.ndarray) and 'has_nan_in_train' (bool).
    """
    oh_metadata = {}
    new_data = None
    new_header = []
    cont_index = [] if index_for_continous else None 
    
    for i in range(x.shape[1]) :
        unique_value_raw = np.unique(x[:,i])
        
        has_nan = np.any(np.isnan(unique_value_raw.astype(float)))
        unique_value = unique_value_raw[~np.isnan(unique_value_raw.astype(float))]
        
        num_cat = len(unique_value) + (1 if has_nan else 0)
        is_binary = set(unique_value).issubset({0, 1})
        
        if (num_cat <= numb_cat) and not is_binary:
            oh_metadata[i] = {
                'categories': unique_value,
                'has_nan_in_train': has_nan
            }
            one_hot, header_one_hot = create_one_hot(x[:,i], unique_value, header[i])
            if has_nan :
                nan_ind = np.isnan(x[:,i].astype(float)).astype(int)
                one_hot = np.column_stack((one_hot, nan_ind))
                header_one_hot.append(f'{header[i]}_nan')
                
            new_data = np.column_stack((new_data, one_hot)) if new_data is not None else one_hot
            new_header.extend(header_one_hot)
        
        else :
            new_data = np.column_stack((new_data, x[:,i])) if new_data is not None else x[:,i].copy()
            new_header.append(header[i])
            if index_for_continous :
                cont_index.append(len(new_header) - 1)
    return new_data, new_header , cont_index, oh_metadata    

def create_one_hot(col, unique_value, header=None, test = False) :
    """
    Helper that builds the indicator columns of one categorical column.
    The last category is dropped and serves as the reference.
 
    Args:
        col (np.ndarray): column of shape (N,).
        unique_value (np.ndarray): categories of the column (without NaN).
        header (str or None): name of the column, used to name the new columns.
        test (bool): if True, no column names are generated. Default is False.
 
    Returns:
        one_hot (np.ndarray): indicator matrix of shape (N, n_categories - 1).
        new_header (list of str): names of the new columns (empty if test is True).
    """
    one_hot = None
    new_header = []
    for i,unique in enumerate(unique_value[:-1]) :
        column = np.where(col == unique, 1, 0)
        
        one_hot = np.column_stack((one_hot, column)) if one_hot is not None else column
        if not test :
            new_header.append(f'{header}_cat{i}') 
        
    return one_hot, new_header

def one_hot_for_test(test, oh_metadata) :
    """
    Apply to a test set the same one-hot encoding that was learned on the training set, so that both sets have identical columns.
 
    Args:
        test (np.ndarray): test data matrix of shape (N_test, D).
        oh_metadata (dict): metadata returned by `one_hot_encoder`.
 
    Returns:
        np.ndarray: encoded test matrix.
    """
    new_test = None
    for i in range(test.shape[1]):
        if i in oh_metadata:
            unique_value = oh_metadata[i]['categories']
            has_nan = oh_metadata[i]['has_nan_in_train']
            
            one_hot, _ = create_one_hot(test[:,i], unique_value, test=True)
            
            if has_nan:
                nan_ind = np.isnan(test[:, i].astype(float)).astype(int)
                one_hot = np.column_stack((one_hot, nan_ind))

            new_test = np.column_stack((new_test, one_hot)) if new_test is not None else one_hot
        
        else:                
            new_test = np.column_stack((new_test, test[:,i])) if new_test is not None else test[:,i]

    return new_test

def compute_loss_mse(y, x, w):
    """
    Compute the mean squared error loss.
 
    Args:
        y (np.ndarray): labels of shape (N,).
        x (np.ndarray): data matrix of shape (N, D).
        w (np.ndarray): weights of shape (D,).
 
    Returns:
        float: value of the loss.
    """
    y_predicted = x.dot(w)
    error = y - y_predicted
    mse = ((1 / (2 * len(y))) * error.T.dot(error))
    return mse

def compute_gradient(y, x, w):
    """Compute the gradient of the MSE loss at w.
 
    Args:
        y (np.ndarray): labels of shape (N,).
        x (np.ndarray): data matrix of shape (N, D).
        w (np.ndarray): weights of shape (D,).
 
    Returns:
        np.ndarray: gradient of shape (D,).
    """
    if np.ndim(y) == 0  :
        N = 1
    else :
        N = y.shape[0]
    error = y - x.dot(w)
    grad_loss = ((-1) / N) * x.T.dot(error)
    return grad_loss

def least_squares(y, x):
    """
    Solve linear regression exactly with the normal equations (X^T X) w = X^T y.
 
    Args:
        y (np.ndarray): labels of shape (N,).
        x (np.ndarray): data matrix of shape (N, D).
 
    Returns:
        w (np.ndarray): optimal weights of shape (D,).
        mse (float): MSE loss of the solution.
    """
    gram_matrix = x.T @ x
    sol = x.T @ y
    w = np.linalg.solve(gram_matrix, sol)
    mse = compute_loss_mse(y, x, w)
    
    return w, mse[0]

def ridge_regression(y, x, lambda_):
    """
    Solve L2-regularized linear regression in closed form.
 
    Args:
        y (np.ndarray): labels of shape (N,).
        x (np.ndarray): data matrix of shape (N, D).
        lambda_ (float): regularization parameter, >= 0.
 
    Returns:
        w (np.ndarray): optimal weights of shape (D,).
        mse (float): unregularized MSE loss on the training data.
    """
    w = np.linalg.solve(x.T @ x + 2*len(y)*lambda_*np.eye(np.shape(x)[1]), x.T @ y)
    mse = compute_loss_mse(y, x, w)
    return w, mse[0]



def mean_squared_error_gd(y, x, initial_w, max_iters, gamma):
    """Train linear regression with full-batch gradient descent (MSE loss).
 
    Args:
        y (np.ndarray): labels of shape (N,).
        x (np.ndarray): data matrix of shape (N, D).
        initial_w (np.ndarray): initial weights of shape (D,).
        max_iters (int): number of iterations.
        gamma (float): step size.
 
    Returns:
        w (np.ndarray): final weights of shape (D,).
        mse (float): final MSE loss.
    """
    w = initial_w.copy()
    for n_iter in range(max_iters):
        #Compute the gradient
        gradient_loss = compute_gradient(y, x, w)
        #update w by gradient
        w = w - gamma*gradient_loss

    mse = compute_loss_mse(y, x, w)
    return w, mse

def mean_squared_error_sgd(y, x, initial_w, max_iters, gamma):
    """Train linear regression with stochastic gradient descent (MSE loss), using one random sample per iteration.
 
    Args:
        y (np.ndarray): labels of shape (N,).
        x (np.ndarray): data matrix of shape (N, D).
        initial_w (np.ndarray): initial weights of shape (D,).
        max_iters (int): number of iterations.
        gamma (float): step size.
 
    Returns:
        w (np.ndarray): final weights of shape (D,).
        mse (float): final MSE loss on the full training set.
    """
    seed = 12
    w = initial_w.copy()

    for n_iter in range(max_iters):
        np.random.seed(seed)
        i = np.random.choice(y.shape[0])
        point_x = x[i,:]
        point_y = y[i]
        gradient_loss = compute_gradient(point_y, point_x, w)

        #update w
        w = w - gamma*gradient_loss
        
    mse = compute_loss_mse(y, x, w)
    return w, mse

def sigma_function(t) :
    """
    Compute the sigmoid function 1 / (1 + exp(-t)).
 
    Args:
        t (float or np.ndarray): input value(s).
 
    Returns:
        float or np.ndarray: values in (0, 1), same shape as t.
    """
    return 1/(1 + np.exp(-t))

def compute_logistic_loss(y, x, w, weighted=None):
    """
    Compute the mean logistic loss log(1 + exp(-y * x w)), optionally with a weight for each sample.
 
    Args:
        y (np.ndarray): labels of shape (N,) or (N, 1), values in {-1, 1}.
        x (np.ndarray): data matrix of shape (N, D).
        w (np.ndarray): weights of shape (D,).
        weighted (np.ndarray or None): sample weights of shape (N,). Default is None.
 
    Returns:
        float: mean logistic loss.
    """
    y = y.squeeze()
    y_pred = x @ w
    loss = np.logaddexp(0, -y * y_pred) #log(1 + exp(-y y_pred))
    return np.mean(loss if weighted is None else weighted*loss)

def ponderate_vector(y) :
    """
    Build balanced class weights: the rarer class gets a larger weight.
    Used to handle imbalanced labels (referred as 'weighted' in the following functions).
 
    Args:
        y (np.ndarray): labels of shape (N,), values in {-1, 1}.
 
    Returns:
        np.ndarray: weight of each sample, shape (N,).
    """
    y_ = y.squeeze()
    N, n_pos = len(y_), np.sum(y_ == 1)
    a = np.where(y_ == 1, N / (2 * n_pos), N / (2 * (N - n_pos))) 
    return a

def gradient_log(y, x, w, lambda_=0, weighted = None) :
    """
    Compute the gradient of the L2-penalized logistic loss. Also works for a single sample (1-D x).
 
    Args:
        y (np.ndarray or float): label(s), values in {-1, 1}.
        x (np.ndarray): data matrix of shape (N, D), or a single sample of shape (D,).
        w (np.ndarray): weights of shape (D,).
        lambda_ (float): regularization parameter. Default is 0.
        weighted (np.ndarray or None): sample weights of shape (N,). Default is None.
 
    Returns:
        np.ndarray: gradient of shape (D,).
    """
    x = np.atleast_2d(x)
    N = x.shape[0]
    weighted = np.ones(N) if weighted is None else weighted
    r = weighted * y * sigma_function(-y * (x@w))
    grad = -(x.T @ r) / N + 2*lambda_*w
    return grad

def hessian_log(y, x, w, lambda_=0, weighted = None) :
    """
    Compute the Hessian of the penalized logistic loss, used by Newton's method.
 
    Args:
        y (np.ndarray): labels of shape (N,), values in {-1, 1}.
        x (np.ndarray): data matrix of shape (N, D).
        w (np.ndarray): weights of shape (D,).
        lambda_ (float): regularization parameter. Default is 0.
        weighted (np.ndarray or None): sample weights of shape (N,). Default is None.
 
    Returns:
        np.ndarray: Hessian matrix of shape (D, D).
    """
    weighted = np.ones(N) if weighted is None else weighted
    N, D = x.shape
    s = sigma_function(x@w)
    p = weighted * s * (1 - s)
    H = (x.T @ (x * p[:,None])) / N + 2 * lambda_ * np.eye(D) #p[:,None] transform shape (N,) to (N,1) equivalent to np.newaxis
    return H

def sgd_logistic_loss_pen(y, x, initial_w, max_iters, gamma, lambda_) :
    """
    Train penalized logistic regression with stochastic gradient descent, using one sample per iteration.
 
    Args:
        y (np.ndarray): labels of shape (N,), values in {-1, 1}.
        x (np.ndarray): data matrix of shape (N, D).
        initial_w (np.ndarray): initial weights of shape (D,).
        max_iters (int): number of iterations.
        gamma (float): step size.
        lambda_ (float): regularization parameter.
 
    Returns:
        w (np.ndarray): final weights of shape (D,).
        loss (float): logistic loss on the full training set.
    """
    seed = 12
    y = y.squeeze()
    w = initial_w.copy()
    
    for n_iter in range(max_iters):
        np.random.seed(seed)
        i = np.random.choice(len(y))
        point_x, point_y = x[i,:], y[i]
            
        gradient_loss = gradient_log(point_y, point_x, w, lambda_)    
        
        #update w
        w = w - gamma*gradient_loss
            
    return w, compute_logistic_loss(y, x, w)

def newton_log_pen(y, x, initial_w, max_iters, lambda_, weighted=None) :
    """
    Train penalized logistic regression with Newton's method, optionally with a weight for each sample.
 
    Args:
        y (np.ndarray): labels of shape (N,), values in {-1, 1}.
        x (np.ndarray): data matrix of shape (N, D).
        initial_w (np.ndarray): initial weights of shape (D,).
        max_iters (int): number of Newton iterations.
        lambda_ (float): regularization parameter.
        weighted (np.ndarray or None): sample weights of shape (N,). Default is None.
 
    Returns:
        w (np.ndarray): final weights of shape (D,).
        loss (float): logistic loss on the training data.
    """
    y = y.squeeze()
    N, D = x.shape
    weighted = np.ones(N) if weighted is None else weighted
    w = initial_w.copy()
    
    for _ in range(max_iters) :
        grad = gradient_log(y, x, w, lambda_, weighted)
        H = hessian_log(y, x, w, lambda_, weighted)
        w = w - np.linalg.solve(H, grad)
        
    return w, compute_logistic_loss(y, x, w)

def threshold_prediction(y_pred, threshold=0) :
    """
    Convert raw scores into class labels: 1 if the score is greater than or equal to 'threshold', otherwise -1.
 
    Args:
        y_pred (np.ndarray): raw scores.
        threshold (float): decision threshold. Default is 0.
 
    Returns:
        np.ndarray: labels in {-1, 1}, same shape as y_pred.
    """
    return np.where(y_pred >= threshold, 1, -1)

def accuracy(y, y_pred):
    """
    Compute the fraction of correct predictions.
 
    Args:
        y (np.ndarray): true labels.
        y_pred (np.ndarray): predicted labels, same shape as y.
 
    Returns:
        float: accuracy in [0, 1].
    """
    return np.mean(y == y_pred)

def f1_score(y, y_pred) :
    """
    Compute the F1 score of the positive class (label 1).
    Returns 0 if there is no true positive.
 
    Args:
        y (np.ndarray): true labels, values in {-1, 1}.
        y_pred (np.ndarray): predicted labels, values in {-1, 1}.
 
    Returns:
        float: F1 score in [0, 1].
    """
    y, y_pred = y.squeeze(), y_pred.squeeze()
    tp = np.sum((y_pred == 1) & (y == 1))
    fp = np.sum((y_pred == 1) & (y == -1))
    fn = np.sum((y_pred == -1) & (y == 1))
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)

## Cross-validation
def build_k_indices(y, k_fold, seed):
    """
    Shuffle the sample indices and split them into k_fold folds of equal size.
    Leftover samples (N modulo k_fold) are dropped.
 
    Args:
        y (np.ndarray): labels of shape (N,).
        k_fold (int): number of folds.
        seed (int): random seed.
 
    Returns:
        np.ndarray: integer array of shape (k_fold, N // k_fold) with the indices of each fold.
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    
    return np.array(k_indices)

def cross_validation_one_iteration(y, x, k_indices, k, lambda_):
    """
    Train ridge regression on all folds except fold k and evaluate it on fold k.
 
    Args:
        y (np.ndarray): labels of shape (N,).
        x (np.ndarray): data matrix of shape (N, D).
        k_indices (np.ndarray): folds returned by `build_k_indices`.
        k (int): index of the test fold.
        lambda_ (float): regularization parameter.
 
    Returns:
        loss_tr (float): RMSE on the training folds.
        loss_te (float): RMSE on the test fold.
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
    """
    Select the best ridge regularization parameter by k-fold cross-validation.
 
    Args:
        y (np.ndarray): labels of shape (N,).
        x (np.ndarray): data matrix of shape (N, D).
        k_fold (int): number of folds.
        lambdas (np.ndarray): values of lambda to test, shape (p,).
 
    Returns:
        best_lambda (float): lambda with the lowest mean test RMSE.
        best_rmse (float): the associated mean test RMSE.
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
    #plot_lambda(np.squeeze(rmse_tr), np.squeeze(rmse_te), lambdas)

    return best_lambda, best_rmse

def cross_validation_one_iteration_log(y, x, k_indices, k, lambda_, gamma):
    """
    Train penalized logistic regression with SGD (30 iterations) on all folds except fold k and evaluate it on fold k.
 
    Args:
        y (np.ndarray): labels of shape (N,), values in {-1, 1}.
        x (np.ndarray): data matrix of shape (N, D).
        k_indices (np.ndarray): folds returned by `build_k_indices`.
        k (int): index of the test fold.
        lambda_ (float): regularization parameter.
        gamma (float): SGD step size.
 
    Returns:
        loss_tr (float): logistic loss on the training folds.
        loss_te (float): logistic loss on the test fold.
    """
    test_index = k_indices[k]
    x_test = x[test_index]
    y_test = y[test_index]
    train_index =  np.delete(k_indices, k, axis=0).ravel()
    x_train = x[train_index]
    y_train = y[train_index]

    w, loss = sgd_logistic_loss_pen(y_train, x_train,  np.zeros(x_train.shape[1]), 30, gamma, lambda_)
    
    loss_tr = loss
    loss_te = compute_logistic_loss(y_test, x_test, w)
    return loss_tr, loss_te

def cross_validation_log(y, x, k_fold, lambdas, gammas):
    """
    Grid search over lambda and gamma for SGD logistic regression using k-fold cross-validation.
 
    Args:
        y (np.ndarray): labels of shape (N,), values in {-1, 1}.
        x (np.ndarray): data matrix of shape (N, D).
        k_fold (int): number of folds.
        lambdas (np.ndarray): values of lambda to test.
        gammas (np.ndarray): values of gamma (step size) to test.
 
    Returns:
        best_lambda (float): best regularization parameter.
        best_gamma (float): best step size.
        best_rmse (float): lowest mean test loss.
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
    #plot_lambda(np.squeeze(rmse_tr[index,:]), np.squeeze(rmse_te[index,:]), gammas)
    #plot_lambda(np.squeeze(rmse_tr[:,np.where(best_gamma == gammas)]), np.squeeze(rmse_te[:, np.where(best_gamma == gammas)]), lambdas)

    return best_lambda, best_gamma, best_rmse

def cross_val_one_log(y, x, k_indices, k, lambda_, weighted=False) :
    """
    Train penalized logistic regression with Newton's method (10 iterations) on all folds except fold k and evaluate it on fold k.
 
    Args:
        y (np.ndarray): labels of shape (N,), values in {-1, 1}.
        x (np.ndarray): data matrix of shape (N, D).
        k_indices (np.ndarray): folds returned by `build_k_indices`.
        k (int): index of the test fold.
        lambda_ (float): regularization parameter.
        weighted (bool): if True, use balanced class weights. Default is False.
 
    Returns:
        loss_tr (float): logistic loss on the training folds.
        loss_te (float): logistic loss on the test fold.
    """
    test_index = k_indices[k]
    x_test = x[test_index]
    y_test = y[test_index]
    train_index =  np.delete(k_indices, k, axis=0).ravel()
    x_train = x[train_index]
    y_train = y[train_index]
    if weighted :
        weight = ponderate_vector(y_train)
    else :
        weight = None

    w, _ = newton_log_pen(y_train, x_train,  np.zeros(x_train.shape[1]), 10, lambda_, weight)
    
    loss_te = compute_logistic_loss(y_test, x_test, w)
    return _, loss_te

def cross_val_new(y, x, k_fold, lambdas, weighted=False) :
    """
    Select the best regularization parameter for Newton logistic regression by k-fold cross-validation.
 
    Args:
        y (np.ndarray): labels of shape (N,), values in {-1, 1}.
        x (np.ndarray): data matrix of shape (N, D).
        k_fold (int): number of folds.
        lambdas (np.ndarray): values of lambda to test.
        weighted (bool): if True, use balanced class weights. Default is False.
 
    Returns:
        best_lambda (float): lambda with the lowest mean test loss.
        best_loss (float): the associated mean test loss.
    """
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
            _, loss_te = cross_val_one_log(y, x, k_indices, k, lambda_, weighted)
            loss_k.append(loss_te)
        log_loss_te.append(np.mean(loss_k))
    
    best_loss = np.min(log_loss_te)
    index = np.argmin(log_loss_te)
    best_lambda = lambdas[index]
    #plot_lambda(np.zeros(len(lambdas)),log_loss_te, lambdas, "lambda")
    
    return best_lambda, best_loss

def cross_validation_threshold(y, x, k_fold, lambda_, thresholds, max_iters=10, weighted=False) :
    """
    Find the decision threshold on the scores that maximizes the mean F1 score across folds.
 
    Args:
        y (np.ndarray): labels of shape (N,), values in {-1, 1}.
        x (np.ndarray): data matrix of shape (N, D).
        k_fold (int): number of folds.
        lambda_ (float): regularization parameter of the Newton logistic regression.
        thresholds (np.ndarray): thresholds to test.
        max_iters (int): number of Newton iterations. Default is 10.
        weighted (bool): if True, use balanced class weights. Default is False.
 
    Returns:
        best_threshold (float): threshold with the highest mean F1 score.
        best_f1 (float): the associated mean F1 score.
    """
    seed = 12
    k_fold = k_fold
    lambda_ = lambda_
    y = y.squeeze()
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    f1 = np.zeros((k_fold, len(thresholds)))
    
    for k in range(k_fold) :
        ind_te = k_indices[k]
        ind_tr = np.delete(k_indices, k, axis=0).ravel()
        weight = None
        if weighted :
            weight = ponderate_vector(y[ind_tr])
        w, _ = newton_log_pen(y[ind_tr], x[ind_tr], np.zeros(x.shape[1]), max_iters, lambda_, weight)
        for i,t in enumerate(thresholds) :
            y_pred = x[ind_te] @ w
            y_pred_class = threshold_prediction(y_pred, t).squeeze()
            f1[k,i] = f1_score(y[ind_te], y_pred_class)
    
    f1_mean = f1.mean(axis=0)
    best = np.argmax(f1_mean)
    #plot_lambda(f1_mean, np.zeros(len(thresholds)), thresholds, 'threshold')
    return thresholds[best], f1_mean[best]

def plot_lambda(loss_tr, loss_te, param, param_name) :
    """
    Plot the train and test errors against a hyperparameter, on a log x-axis.
 
    Args:
        loss_tr (array-like): training errors, same length as param.
        loss_te (array-like): test errors, same length as param.
        param (array-like): values of the hyperparameter.
        param_name (str): label of the x-axis.
 
    Returns:
        None: creates a matplotlib figure.
    """
    plt.figure()
    plt.semilogx(param, loss_tr, marker=".", color="b", label="train error")
    plt.semilogx(param, loss_te, marker=".", color="r", label="test error")
    plt.xlabel(param_name)
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    

def create_csv_submission(ids, y_pred, name):
    """
    Write a prediction file (columns 'Id' and 'Prediction') in the format
    required for an AIcrowd submission.
 
    Args:
        ids (list or np.ndarray): sample ids.
        y_pred (list or np.ndarray): predictions, only values -1 or 1.
        name (str): path of the csv file to create.
 
    Returns:
        None: writes the file.
 
    Raises:
        ValueError: if y_pred contains a value other than -1 or 1.
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
            










        
        
