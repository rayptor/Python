import numpy as np

# On a general class of optimal order multipoint methods for solving nonlinear equations
def SharmaArgyrosKumar(
    f: callable,
    df: callable,
    x0: np.float128,
    k:int = 10,
    tol = np.finfo(np.float128).eps
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        dfxn = df(xn)
        det1 = 1.0
        delta1 = df(xn)
        wn = np.float128(xo - det1 / delta1 * fxn)
        fwn = f(wn)
        fxw = (fwn - fxn) / (wn - xn)
        det2 = 1.0 * fxn - 1.0 * fwn
        delta2 = dfxn * fwn - fxw * fxn

        zn = xn - det2 / delta2 * fxn
        fzn = f(zn)
        fxz = (fzn - fxn) / (zn - xn)
        m3 = np.array([[1.0, fxn, xn * fxn],
				[1.0, fwn, wn * fwn],
				[1.0, fzn, zn * fzn]], dtype=np.float64)
        det3 = np.linalg.det(m3)
        md3 = np.array([[dfxn, fxn, xn * fxn],
				[fxw, fwn, wn * fwn],
				[fxz, fzn, zn * fzn]], dtype=np.float64)
        delta3 = np.linalg.det(md3)

        un = xn - det3 / delta3 * fxn
        fun = f(un)
        fxu = (fun - fxn) / (un - xn)
        m4 = np.array([[1.0, fxn, xn * fxn, xn**2 * fxn],
				[1.0, fwn, wn * fwn, wn**2 * fwn],
				[1.0, fzn, zn * fzn, zn**2 * fzn],
				[1.0, fun, un * fun, un**2 * fun]], dtype=np.float64)
        det4 = np.linalg.det(m4)
        md4 = np.array([[dfxn, fxn, xn * fxn, xn**2 * fxn],
				[fxw, fwn, wn * fwn, wn**2 * fwn],
				[fxz, fzn, zn * fzn, zn**2 * fzn],
				[fxu, fun, un * fun, un**2 * fun]], dtype=np.float64)
        delta4 = np.linalg.det(md4)
        xn = xn - det4 / delta4 * fxn
        
        if np.less(np.fabs(xn - xo), tol):
          break
        else:
        	xo = xn
    
    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# x ≈ 1.05479303061959275440578962...
# x = 1.054793030619715299
#     1.054793030619933569
#     1.054793030619591728 1.95
#     1.054793079658612225 1.1
#     1.054793030581218613 2
if __name__ == "__main__":
    # x^45 - 4exp(x) + sin(3x) - xcos(x) + 1
    f_SharmaArgyrosKumar = lambda x: x**45 - 4.0*np.exp(x) + np.sin(3.0*x) - x*np.cos(x) + 1.0
    df_SharmaArgyrosKumar = lambda x: 45.0*x**44 - 4.0*np.exp(x) + x*np.sin(x) - np.cos(x) + 3.0*np.cos(3.0*x)
    print("x =", SharmaArgyrosKumar(f_SharmaArgyrosKumar, df_SharmaArgyrosKumar, 1.2))
