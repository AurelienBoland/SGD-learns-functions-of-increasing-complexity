import numpy as np

def est_density(X, Y, Z=None, n_classes=2): 
    assert max(X) < n_classes and max(Y) < n_classes and (Z is None or max(Z) < n_classes), "X, Y, Z should be in [0, n_classes-1]"
    if Z is None:
        return _est_density_2D(X, Y, n_classes)
    X = X.astype(int)
    Y = Y.astype(int)
    Z = Z.astype(int)
    n = len(X)
    p = np.zeros((n_classes, n_classes, n_classes)) 
    for i in range(n):
        p[X[i], Y[i], Z[i]] += 1.0
    p /= n
    return p

def _est_density_2D(X, Y, n_classes):
    X = X.astype(int)
    Y = Y.astype(int)
    n = len(X)
    p = np.zeros((n_classes, n_classes))
    for i in range(n):
        p[X[i], Y[i]] += 1.0
    p /= n
    return p

def I_XY(p:np.ndarray, idx=(0,1)) -> float:
    """
    Compute I(X, Y) for joint density p[x, y, z]
    """
    if len(p.shape) == 2:
        n_classes = p.shape[0]
        p = np.append(p[:,:,np.newaxis], np.zeros((n_classes,n_classes,n_classes-1)), axis=2)
    exlude = (0+1+2) - np.sum(idx)
    p_ab = np.sum(p, axis=exlude)
    p_a = np.sum(p_ab, axis=1, keepdims=True)
    p_b = np.sum(p_ab, axis=0, keepdims=True)
    I = np.sum(p_ab * np.nan_to_num(np.log2( p_ab / (p_a * p_b) )))
    return I

def I_XY_Z(p:np.ndarray) -> float:
    """
    Compute I(X, Y | Z) for joint density p[x, y, z]
    """
    pz = np.sum(p, axis=(0,1), keepdims=True) 
    p_xy_z = p / pz 
    p_x_z =  np.sum(p, axis=1, keepdims=True) / pz  
    p_y_z =  np.sum(p, axis=0, keepdims=True) / pz 
    I = np.sum(p * np.nan_to_num(np.log2( p_xy_z / (p_x_z * p_y_z) )))
    return I

def mu_metric(p) -> float:
    """
    Compute mu performance metric from the paper
    p: joint density p[model_ouptut, submodel_output, target]
    """
    return I_XY(p) - I_XY_Z(p)