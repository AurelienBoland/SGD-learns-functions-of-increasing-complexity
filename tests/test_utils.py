import pytest
import numpy as np

from sgd_complexity.utils import (
    est_density,
    I_XY,
)

# remark:
# H_Y_XZ is not tested because it is to much work to compute each term mamually

X = np.array([0,1,1,1] + [0,1,1,1] + [0,0,0,0] + [1,1,1,1])
Y = np.array([0,0,1,1] + [0,0,1,1] + [0,0,1,1] + [0,0,1,1]) 
Z = np.array([0,0,0,1] + [0,0,0,1] + [0,1,0,1] + [0,1,0,1])

@pytest.fixture(scope="session")
def p():
    return est_density(X, Y, Z)

def test_est_density(p):
    q = np.ones((2,2,2)) * 1/16
    q[0,0,0] = 3/16
    q[1,0,0] = 3/16
    q[1,1,0] = 3/16
    q[1,1,1] = 3/16
    assert np.array_equal(p,q)

def test_I_XY_2D(p):
    p2 = np.zeros((2,2))
    p2[0,0] = p[0,0,0] + p[0,0,1]
    p2[0,1] = p[0,1,0] + p[0,1,1]
    p2[1,0] = p[1,0,0] + p[1,0,1]
    p2[1,1] = p[1,1,0] + p[1,1,1]
    assert I_XY(p2) == I_XY(p)
    