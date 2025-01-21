import numpy as np

def Thukral(
  f: callable,
  x0: np.float128,
  k:int = 10,
  tol = np.finfo(np.float128).eps
    ) -> np.float128:
  dec = np.finfo(np.float128).precision
  xo = x0
  xn = xo
  for _ in range(k):
    fxn = np.float128(f(xn))
    if np.isclose(fxn, 0.0, atol=tol) == True:
        break
    wn = xn + fxn
    fwn = f(wn)
    fwx = (fwn - fxn) / (wn - xn)
    yn = xo - (fxn / fwx)
    fyn = np.float128(f(yn))
    fxy = (fxn - fyn) / (xn - yn)
    fyw = (fyn - fwn) / (yn - wn)
    fxw = (fxn - fwn) / (xn - wn)
    phi3 = fxw / fyw
    zn = yn - phi3 * (fyn / fxy)
    fzn = f(zn)
    u2 = fzn / fwn
    u3 = fyn / fxn
    u4 = fyn / fwn
    fyz = (fyn - fzn) / (yn - zn)
    fxz = (fxn - fzn) / (xn - zn)
    eta = (1 / (1 + 2 * u3 * u4**2)) / (1 - u2)
    an = zn - eta * (fzn / (fyz - fxy + fxz))
    fan = f(an)
    u1 = fzn / fxn
    u5 = fan / fxn
    u6 = fan / fwn
    sigma = 1 + u1 * u2 - u1 * u3 * u4**2 + u5 + u6 \
        + u1**2 * u4 + u2**2 * u3 + 3 * u1 * u4**2 \
        * (u3**2 - u4**2) * (1 / fxy)
    fya = (fyn - fan) / (yn - an)
    fza = (fzn - fan) / (zn - an)
    xn = zn - sigma * (fyz * fan) / (fya * fza)

    if np.less(np.fabs(xn - xo), tol):
      break
    else:
      xo = xn
  
  return np.format_float_positional(np.float128(xn), \
      unique=False, precision=dec)

if __name__ == "__main__":
  # 2x^9 - sqrt(17pi) sin(2/3x) - 16xcos(2x) - 64 = 0
  f = lambda x: 2*x**9 - np.sqrt(17*np.pi)*np.sin(2/3*x) - 16*x*np.cos(2*x) - 64
  print("x =", Thukral(f, 1.4))
