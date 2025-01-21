import numpy as np

def WangLiu(
    f: callable,
    df: callable,
    x0: np.float128,
    k: int = 10,
    tol = np.finfo(np.float128).eps
    ) -> np.float128:

    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo

    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn = df(xn)
        yn = xo - fxn/dfxn
        fyn = f(yn)
        fxy = (fyn-fxn)/(yn-xn)
        zn = yn - f(yn)/(2.0*fxy-dfxn)
        fzn = f(zn)
        fxz = np.float128((fzn-fxn)/(zn-xn))
        fyz = np.float128((fzn-fyn)/(zn-yn))
        fyxx = np.float128((fxy-dfxn)/(yn-xn))
        xn = zn - fzn / (2.0*fxz+fyz-2*fxy+(yn-zn)*fyxx)

        if np.less(np.fabs(xn-xo), tol) == True:
            break
        else:
            xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

if __name__ == "__main__":
    #x ≈ 1.30361055802424...
    f = lambda x: x**14 + 4.0*x**2 \
    		- x*np.exp(x**7 - 2*x) + 10.0
    df = lambda x: 14*x**13 + 8*x \
    		- np.exp(x**7 - 2*x)*(7*x**6 - 2)
    print("x =", WangLiu(f, df, 1.1))
