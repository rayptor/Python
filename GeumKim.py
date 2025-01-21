import numpy as np
    
# Computing Simple Roots by an Optimal Sixteenth-Order Class
def GeumKimY1(
    f: callable,
    df: callable,
    x0: np.float128,
    k:int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:

    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    beta = 2
    sigma = -beta

    for _ in range(k):
        fxn = f(xn)
        dfxn = df(xn)
        yn = xo - fxn/dfxn
        fyn = f(yn)
        un = fyn/fxn
        kf = (1 + beta * un + (-9 + ((5 * beta) / 2)) * un**2) \
            / (1 + (beta - 2) * un + (-4 + beta  /2) * un**2)
        zn = yn - kf * (fyn / dfxn)
        fzn = f(zn)
        vn = fzn / fyn
        wn = fzn / fxn
        hf = (1 + 2 * un + (2 + sigma) * wn) / (1 - vn + sigma * wn)
        sn = zn - hf * (fzn / dfxn)
        fsn = f(sn)
        tn = fsn / fzn
        phi1 = 11 * beta**2 - 66 * beta + 136
        phi2 = 2 * un * (sigma**2 - 2 * sigma - 9) - 4 * sigma - 6
        guw = (-1 / 2) * (un * wn * (6 + 12 * un + un**2 + un**2 \
            * (24 - 11 * beta) + un**3 * phi1 + 4 * sigma))+phi2 * wn**2
        wf = (1 + 2 * un + (2 + sigma) * vn * wn) / (1 - vn - 2 * wn \
            - tn + 2 * (1 + sigma) * vn * wn) + guw
        xn = sn - wf * (fsn / dfxn)

        if np.less(np.fabs(xn - xo), tol) == True:
            break
        else:
            xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

    # x ≈ 1.23006617693139398830483636...
    # x = 1.230066176931393862
if __name__ == "__main__":
    f = lambda x: 13*x**9 - x * np.exp(x**7) + np.cos(x) + np.sqrt(8)/x**3 + x**2
    df = lambda x: 117 * x**8 - np.exp(x**7) * (7 * x**7 + 1) - (6 * np.sqrt(2)) / x**4 - np.sin(x) + 2 * x
    print("x =", GeumKimY1(f, df, 1.3))
