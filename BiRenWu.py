import numpy as np

def BiRenWu_equ_40(
    f: callable,
    df: callable,
    x0: np.float128,
    k: int = 10,
    tol = np.finfo(np.float128).eps
    ) -> np.float128:

    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    alpha = 1.0

    for _ in range(k):
        fxn = f(xn)
        dfxn = df(xn)
        yn = xo - fxn/dfxn
        fyn = f(yn)
        zn = yn  - ((2 * fxn - fyn) / (2 * fxn - 5 * fyn)) \
            * (fyn / dfxn)
        fzn = f(zn)
        fzx = np.float128((fzn - fxn) / (zn - xn))
        fzy = np.float128((fzn - fyn) / (zn - yn))
        fzxx = np.float128((fzx - dfxn) / (zn - xn))
        xn1 = np.power((fxn / (fxn - alpha * fzn)), 2 / alpha)
        xn2 = fzn / (fzy + fzxx*(zn - yn))
        xn = zn - xn1 * xn2

        if np.less(np.fabs(xn-xo), tol) == True:
            break
        else:
            xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

if __name__ == "__main__":
	f = lambda x: x**6 - np.exp(x**3) - 1 / np.sqrt(5+x) \
		+ 8 * x + 100
	df = lambda x: 6 * x**5 - 3 * np.exp(x**3) * x**2 \
		+ 1 / (2 * (5 + x)**(3/2)) + 8
	print("x =", BiRenWu_equ_40(f, df, 1.7))
