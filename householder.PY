import numpy as np

def qr_householder(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = m.shape[0]
    Q = np.identity(n)
    R = np.copy(m)
    for i in range(n-1):
        x = R[i:,i]
        e = np.zeros_like(x)
        e[0] = 1
        v = x + np.copysign(np.linalg.norm(x), x[0]) * e
        H = np.identity(n)
        H[i:,i:] -= ((2.0 * np.outer(v,v.T)) / np.inner(v.T,v))
        R = H @ R
        Q = Q @ H

    return Q,R

a = np.array([[2,-1,3],[2,-5,-3],[1,3,2]])
# A = np.array([[2,5,0,-1],[-3,5,11,9],[8,5,-2,1],[7,-6,4,-8]])
Q,R = qr_householder(a)
print('Q Householder :\n', np.around(Q, decimals=8))
print('R Householder :\n', np.around(R, decimals=8))
print('QR Householder :\n', Q@R)
