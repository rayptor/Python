############################################################################################################
##                                                                                                        ##
## MATHÉMATIQUES : ANALYSE NUMÉRIQUE                                                                      ##
##                                                                                                        ##
## Implémentation de la méthode suivante :                                                                ##
## "A new class of optimal four-point methods with convergence order 16 for solving nonlinear equations"  ##
## publiée en 2014 par Somayeh Sharifia, Mehdi Salimi, Stefan Siegmund, Taher Lotfic                      ##
## pour la résolution des équations non linéaires univariées en ne faisant appel qu'à la dérivée première ##
##                                                                                                        ##
## Article scientifique au format PDF disponible à : https://arxiv.org/abs/1410.2633                      ##
##                                                                                                        ##
############################################################################################################

import numpy as np
import sys

def Solve(
    f: None,
    df: None,
    x0: np.longdouble,
    k:int = 10,
    tol: np.longdouble = 1e-15
    ) -> np.longdouble:
    dec = np.finfo(np.longdouble).precision
    xo = x0
    xn = xo
    if f == None or df == None:
        raise RuntimeError("Aucune fonction n'a été passée en argument.")
    if tol <= 0.0:
        raise ValueError("L'epsilon doit être strictement positif.")
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn = df(xn)
        yn = np.longdouble(xo - fxn/dfxn)
        fyn = f(yn)
        try:
            tn = np.longdouble(fyn/fxn)
            zn = np.longdouble(yn - ((1.0+tn**2) * (1.0+2.0*tn+2.0*tn**2) \
                + tn**2*(2.0-8.0*tn-2.0*tn**2)) * fyn/dfxn)
            fzn = f(zn)
            un = np.longdouble(fzn / fxn)
            sn = np.longdouble(fzn / fyn)
            wn = np.longdouble(zn - (4.0*un-5.0*sn+(6.0+sn**3) \
                * (tn**2+sn) + (1.0+un**3)*(1.0+2.0*tn)) * fzn/dfxn)
            fwn = f(wn)
            pn = np.longdouble(fwn / fxn)
            qn = np.longdouble(fwn / fyn)
            rn = np.longdouble(fwn / fzn)
        except ZeroDivisionError:
            print("fyn/fxn !")
            sys.exit(1)
        xn1 = np.longdouble((1.0+tn) * (2.0*tn+tn**3) + 4.0*tn**2 \
            - tn**3 - tn**4 - 2.0*sn**2)
        xn2 = np.longdouble(2.0*tn*rn + 2.0*sn*un + 24.0*tn**4 + tn*un)
        if (qn < (1.0-tol)) or (qn > (1.0-tol)) \
           (rn < (1.0-tol)) or (rn > (1.0-tol)) :
            xn3 = np.longdouble((2.0*tn**3-10.0*tn*un**2+6.0*tn**2*un) \
                / (1.0+2.0*tn*un) + ((1.0+2.0*pn+2.0*qn)/(1.0-rn)) \
                + (6.0*pn)/(1.0+qn))
        if (un < (1.0-tol)) or (un > (1.0-tol)) \
           (tn < (1.0-tol)) or (tn > (1.0-tol)) :
            xn4 = np.longdouble((2.0*un+6*un**2)/(1.0+un) + (sn+2.0*sn**2) \
                / (1.0+sn**2) + (6.0*tn**2*rn+6.0*tn**3*rn-4.0*sn**2*un) \
                / (1.0+tn))
        xn = np.longdouble(wn - (xn1-xn2+xn3-xn4) * fwn/dfxn)
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn
    else:
        raise ValueError(f"Aucune solution convenable n'a été trouvée durant ces {k} itérations.")

    return np.format_float_positional(np.longdouble(xn), unique=False, precision=dec)

f = lambda x: x**7-np.sqrt(np.pi)*(2.0*x**2-4.0*x+2.0)+np.exp(x)-1.0
df = lambda x: 7.0*x**6+ np.exp(x) - 4.0*np.sqrt(np.pi)*(x-1.0)
print("x^7-np.sqrt(pi)*(2*x^2-4*x+2)+np.exp(x)-1, x =", Solve(f, df, 0.5))
# x^7-np.sqrt(pi)*(2*x^2-4*x+2)+np.exp(x)-1, x = 0.544025337130809784