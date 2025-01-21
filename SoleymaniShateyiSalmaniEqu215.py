import numpy as np
    
# Computing Simple Roots by an Optimal Sixteenth-Order Class
def SoleymaniShateyiSalmaniEqu215(
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
        yn = xo - fxn / dfxn
        fyn = f(yn)
        zn = yn - fxn / (fxn - 2 * fyn) * (fyn / dfxn)
        fzn = f(zn)
        dfzn = df(zn)

        fxy = np.float128((fyn - fxn) / (yn - xn))
        fyz = np.float128((fzn - fyn) / (zn - yn))
        fyx = np.float128((fxn - fyn) / (xn - yn))
        fxz = np.float128((fzn - fxn) / (zn - xn))
        fzx = np.float128((fxn - fzn) / (xn - zn))

        wn_f = np.float128((fxy * fzn) / (fyz * fxz))
        wn = np.float128(zn - (1.0 + fzn / fxn) * wn_f)
        fwn = f(wn)
        fwx = np.float128((fxn - fwn) / (xn - wn))
        fwxx = np.float128((fwx - dfxn) / (wn - xn))

        y = yn - xn
        z = zn - xn
        w = wn - xn

        b5_1 = fzx * w * y * (wn - yn)
        b5_2 = fwx * (yn - zn) * y + fyx * (zn - wn) * w
        b5_3 = (wn - yn) * (wn - zn) * (yn - zn) * dfzn
        b5 = b5_1 + z * (b5_2 - b5_3)
        b2 = dfxn + fxn * b5

        b4_1 = (zn - xn) * fyx + (xn - yn) * fzx
        b4_2 = (zn - xn) * fyn + (xn - yn) * fzn
        b4 = b4_1 + (yn - zn) * b2 + b4_2 * b5
        b3 = fwxx + fwx * b5 - (wn - xn) * b4

        xn_num = (1 + b5 * (wn - xn)**2) * fwn
        xn_den1 = dfxn + 2 * b3 * (wn - xn)
        xn_den2 = (3 * b4 + b3 * b5) * (wn - xn)**2
        xn_den3 = 2 * b4 * b5 * (wn - xn)**3
        xn_den = xn_den1 + xn_den2 + xn_den3
        xn = wn - xn_num / xn_den

        if np.less(np.fabs(xn - xo), tol) == True:
            break
        else:
            xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

    # x ≈ 1.61350763512373435597073879...
if __name__ == "__main__":
    f = lambda x: x**13 - np.log(2*x - 1) - x*np.sqrt(5) + np.exp(x) * 100 - 1000
    df = lambda x: 13*x**12 + 100 * np.exp(x) - 2 / (2*x - 1) - np.sqrt(5)
    print("x = ", SoleymaniShateyiSalmaniEqu215(f, df, 1.6))
