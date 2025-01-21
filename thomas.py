import numpy as np

def thomas(
    c: np.ndarray,
    d: np.ndarray,
    e: np.ndarray,
    b: np.ndarray
    ) -> np.ndarray:
    
    n = d.shape[0]
    r = np.zeros(n, dtype=np.float64)
    s = np.zeros_like(r)
    x = np.zeros_like(r)

    r[0] = d[0]
    s[0] = b[0]/r[0]

    for i in range(1,n):
        r[i] = d[i] - (c[i-1] * e[i-1] / r[i-1])
        s[i] = (b[i] - c[i-1] * s[i-1]) / r[i]

    x[n-1] = s[n-1]
    for i in range(n-2,-1,-1):
        x[i] = s[i] - e[i] * x[i+1] / r[i]

    return x

if __name__ == "__main__":
	c = np.array([1,4,-2,3])
	d = np.array([3,-1,2,5,4])
	e = np.array([1,4,-2,3])
	b = np.array([-20,0,10,-18,-15])

print("Thomas(x) =", thomas(c,d,e,b)) # -7, 1, 2, -1, -3