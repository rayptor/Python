import numpy as np
    
def TaoMadhu(
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
        dfxn = df(xn)
        psi2 = xo - fxn / dfxn
        fpsi2 = f(psi2)
        fpsi2xx = (fpsi2 - dfxn) / (fpsi2 - xn)
        psi4 = psi2 - fpsi2 / (dfxn + (2 * fpsi2xx) * (psi2 - xn))
        fpsi4 = f(psi4)
        fpsi4xx = (fpsi4 - dfxn) / (fpsi4 - xn)

        b2_1 = fpsi2xx * (psi4 - xn)
        b2_2 = fpsi4xx * (psi2 - xn)
        b2 = np.float128((b2_1 - b2_2) / (psi4 - psi2))
        b3 = np.float128((fpsi4xx - fpsi2xx) / (psi4 - psi2))

        psi8_1 = dfxn + 2 * b2 * (psi4 - xn)
        psi8_2 = 3 * b3 * (psi4 - xn)**2
        psi8 = psi4 - fpsi4 / (psi8_1 + psi8_2)
        fpsi8 = f(psi8)
        fpsi8xx = (fpsi8 - dfxn) / (fpsi8 - xn)

        s1, s2, s3 = (psi2 - xn), (psi4 - xn), (psi8 - xn)
        s1s1, s2s2, s3s3 = s1 * s1, s2 * s2, s3 * s3
        d1 = -s1s1 * s2 + s1 * s2s2 + s1s1 * s3
        d2 = s2s2 * s3 - s1 * s3s3 + s2 * s3s3
        d = (d1 - d2)

        c2_1_1 = fpsi2xx * (-s2s2 * s3 + s2 * s3s3)
        c2_1_2 = fpsi4xx * (s1s1 * s3 - s1 * s3s3)
        c2_1_3 = fpsi8xx * (-s1s1 * s2 - s1 * s2s2)
        c2_1 = (c2_1_1 + c2_1_2 + c2_1_3)
        c2_2 = d
        c2 = c2_1 / c2_2

        c3_1_1 = fpsi2xx * (s2s2 - s3s3)
        c3_1_2 = fpsi4xx * (-s1s1 + s3s3)
        c3_1_3 = fpsi8xx * (s1s1 - s2s2)
        c3_1 = (c3_1_1 + c3_1_2 + c3_1_3)
        c3_2 = d
        c3 = c3_1 / c3_2

        c4_1_1 = fpsi2xx * (-s2 + s3)
        c4_1_2 = fpsi4xx * (s1 - s3)
        c4_1_3 = fpsi8xx * (-s1 + s2)
        c4_1 = (c4_1_1 + c4_1_2 + c4_1_3)
        c4_2 = d
        c4 = c4_1 / c4_2

        psi16_1 = 2 * c2 * s3
        psi16_2 = 3 * c3 * s3s3
        psi16_3 = 4 * c4 * s3s3 * s3
        psi16 = psi8 - fpsi8 / (dfxn + psi16_1 + psi16_2 + psi16_3)
        xn = psi16

        if np.less(np.fabs(xn-xo), tol) == True:
            break
        else:
            xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

    # x ≈ 1.27311032510007475756096131...
if __name__ == "__main__":
    # x^11 - 4xlog(x^5) - sqrt(x+2) - 2pi
    f = lambda x: x**11 - 4 * x * np.log(x**5) - np.sqrt(x + 2) - 2 * np.pi
    df = lambda x: 11 * x**10 - 4 * np.log(x**5) - 1 / (2 * np.sqrt(x + 2)) - 20
    print("x =", TaoMadhu(f, df, 1.2))
