## SOURCE: https://github.com/oyamad/simplex_grid
"""
Filename: simplex_grid.py
Author: Daisuke Oyama
Thie module provides a function that constructs a grid for a simplex as
well as one that determines the index of a point in the simplex.
"""
import numpy as np
import scipy.special
from numba import jit


def simplex_grid(m, n=10):
    r"""
    Construct an array consisting of the integer points in the
    (m-1)-dimensional simplex :math:`\{x \mid x_0 + \cdots + x_{m-1} = n
    \}`, or equivalently, the m-part compositions of n, which are listed
    in lexicographic order. The total number of the points (hence the
    length of the output array) is L = (n+m-1)!/(n!*(m-1)!) (i.e.,
    (n+m-1) choose (m-1)).
    Parameters
    ----------
    m : scalar(int)
        Dimension of each point. Must be a positive integer.
    n : scalar(int)
        Number which the coordinates of each point sum to. Must be a
        nonnegative integer.
    Returns
    -------
    out : ndarray(int, ndim=2)
        Array of shape (L, m) containing the integer points in the
        simplex, aligned in lexicographic order.
    Notes
    -----
    A grid of the (m-1)-dimensional *unit* simplex with n subdivisions
    along each dimension can be obtained by `simplex_grid(m, n) / n`.
    Examples
    --------
    >>> simplex_grid(3, 4)
    array([[0, 0, 4],
           [0, 1, 3],
           [0, 2, 2],
           [0, 3, 1],
           [0, 4, 0],
           [1, 0, 3],
           [1, 1, 2],
           [1, 2, 1],
           [1, 3, 0],
           [2, 0, 2],
           [2, 1, 1],
           [2, 2, 0],
           [3, 0, 1],
           [3, 1, 0],
           [4, 0, 0]])
    >>> simplex_grid(3, 4) / 4
    array([[ 0.  ,  0.  ,  1.  ],
           [ 0.  ,  0.25,  0.75],
           [ 0.  ,  0.5 ,  0.5 ],
           [ 0.  ,  0.75,  0.25],
           [ 0.  ,  1.  ,  0.  ],
           [ 0.25,  0.  ,  0.75],
           [ 0.25,  0.25,  0.5 ],
           [ 0.25,  0.5 ,  0.25],
           [ 0.25,  0.75,  0.  ],
           [ 0.5 ,  0.  ,  0.5 ],
           [ 0.5 ,  0.25,  0.25],
           [ 0.5 ,  0.5 ,  0.  ],
           [ 0.75,  0.  ,  0.25],
           [ 0.75,  0.25,  0.  ],
           [ 1.  ,  0.  ,  0.  ]])
    References
    ----------
    A. Nijenhuis and H. S. Wilf, Combinatorial Algorithms, Chapter 5,
    Academic Press, 1978.
    """
    L = num_compositions(m, n)
    out = np.empty((L, m), dtype=int)

    x = np.zeros(m, dtype=int)
    x[m-1] = n

    for j in range(m):
        out[0, j] = x[j]

    h = m

    for i in range(1, L):
        h -= 1

        val = x[h]
        x[h] = 0
        x[m-1] = val - 1
        x[h-1] += 1

        for j in range(m):
            out[i, j] = x[j]

        if val != 1:
            h = m

    return out
def num_compositions(m, n):
    """
    The total number of m-part compositions of n, which is equal to
    (n+m-1) choose (m-1).
    Parameters
    ----------
    m : scalar(int)
        Number of parts of composition.
    n : scalar(int)
        Integer to decompose.
    Returns
    -------
    scalar(int)
        Total number of m-part compositions of n.
    """
    # docs.scipy.org/doc/scipy/reference/generated/scipy.misc.comb.html
    return scipy.special.comb(n+m-1, m-1, exact=True)


