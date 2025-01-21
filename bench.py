# /usr/local/bin/python3.11-intel64 -m pip install --upgrade numpy, sympy, scipy, mpmapth, numexpr, matplotlib, image
# https://cdiese.fr/python-pip/
# pip freeze --local | awk -F "=" '{print "pip install -U "$1}' | sh

# from mpmath import *
# mp.dps = 1000000
# mp.findroot(lambda x: mp.sin(x) * mp.power((mp.power(x, (mp.power(x, 2))) + mp.power(x - 1, 2)), 2), 3.14)
# sin(x)*(x**(x**2)+(x-1)**2)**2 = 3.141592653589792...
# x^10-2x^9+9x^8-5x^7+3x^6-7x^5+6x^4-x^3+8x^2-x+1=0

# try:
#     (exécuter le code)
# except:
#     (s'il y a une exception, la traiter)
# else:
#     (s'il n'y pas d'exception exécuter ce code)
# finally:
#     (toujours exécuter ce code)

# 1+x*log(x)+(2-x)log(1-x)=0 > x≈0.31438348556347124201
# exp((x-1)(x-2)(x-3))-2=0 > x≈3.24728666913069
# x^11-2cos(x^7-x^6+x^2-1)-x+5=0 > x≈-1.1662383633886558404
# x^4+log(x^2-1)+atan(x*0.2)-x*cos(x)-exp(x^9-x^2-x+5)+10=0 > x ≈ -1.10933721739358...

# // 1/x^4-exp(1000/x)+2100, x≈130.72420673480630199
# // exp(-x) - pow(x,x) + x*sin(pow(x,4))+0.25, x ≈ 1.25229472118419...
# // pow(x,26) * pow(log(x),2) + exp(x*sin(x)) - 10.0; // x ≈ 1.21951339224370...
# // pow(x,5)*sin(x)-x*exp(x); // x ≈ 1.13347641516906...
# // exp(-x) - pow(x,x) + x*sin(pow(x,4))+0.25; // x ≈ 1.25229472118419...
# // pow(x,25) + sin(2.0*x - 1.0) - cos(x) - 1.0; x ≈ 0.987151371179975...

# multigrille
# https://blog.csdn.net/zry1318/article/details/103713546?spm=1001.2101.3001.6650.18&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-18-103713546-blog-102820852.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-18-103713546-blog-102820852.pc_relevant_default

from math import sqrt, fabs, acos, asin, atan, cos, sin, tan, exp, log, pi, factorial, isclose
import numpy as np
import mpmath as mm
from numpy.polynomial import Polynomial as Poly
# from python import numpy as np # -> pour Codon
import scipy.io as spio
#from scipy.sparse import spdiags, coo
from random import *
import sys
import os
from typing import Callable, Generic, TypeVar, Union
import time as time

# import matplotlib
# matplotlib.use('pgf')
import matplotlib as mpl
import matplotlib.pyplot as plt
# from matplotlib.ticker import autoMinorLocator, FormatStrFormatter

    # x = np.linspace(0,1,1000)
    # y = x+np.exp(x)-x*np.sin(x-1.0/3.0)+x*np.sqrt(5.0)-3.0
    # fig = plt.figure()
    # ax = fig.gca()
    # ax.xaxis.set_minor_locator(autoMinorLocator())
    # ax.yaxis.set_minor_locator(autoMinorLocator())
    # ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    # ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    # plt.plot(x,y, 'k-', label = r'$y=x+np.exp(x)-x*np.sin(x-1.0/3.0)+x*np.sqrt(5.0)-3.0$')
    # plt.axhline(0, color='k')
    # plt.axvline(0, color='k')
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    # plt.minorticks_on()
    # # plt.xlabel("x")
    # # plt.ylabel("y")
    # plt.title(f'x+np.exp(x)-x*np.sin(x-1.0/3.0)+x*np.sqrt(5.0)-3.0')
    # plt.show()
    # exit()

# MaTRICE :
# ---------
# dimension, réelle hermitienne, symétrique ou non, triangulaire ou non
# transposée, adnointe, inverse, déterminant, rang, trace, norme euclidienne,
# polynôme caractéristique, valeurs propres, vecteurs propres,
# diagonalisation, nombre de conditionnement, factorisation LU, factorisation
# de Cholesky, décomposition QR, Schur, SVD et de Jordan, espace nul et sa base
# classic product, Kronecker product, Strassen product, Faddeev-Leverier, Jacobi,
# Householder, Givens, Gauss Elimination, LU, Cholesky, Thomas, Gauss-Seydel, Sor,
# cg, pcg-ic, cgs, tfqmr, bicgstab, bicgstab(l), qmrcgstab, gp-bicg, tfqmors

# class Matrix:
#     def __init__(self, xnw):
#         self.matrix = self.new(self, n)
#     def __add__(self, M):
#     def __sub__(self, M):
#     def __mul__(self, M):
#     def __mulb__(self, M):
#     def __mulk__(self, M):
#     def __div__(self, M, s):
#     def __repr__(self):
#     def get(i, j):
#     def set(i, j, v):
#     def identity(n):
#     def eigenvalues(): #JaCOBI
#     def eigenvectors():
#     def PolCarac():
#     def Power():
#     def PowerInv():
#     def det():
#     def rank():
#     def QR_GSM():
#     def Householder():
#     def inverseTriU():
#     def inverseTriL():
#     def inverse():
#     def solveGaussElim():
#     def solveLU():
#     def solveTriL():
#     def solveTriU():
#     def solveThomas():
#     def solveGaussSeydel():
#     def solveSOR():
#     def solveCG():
#     def solvePCG_Diag():
#     def solvePCG_IC():
#     def solveBICR():
#     def solveCGS():
#     def solveBICGSTaB():
#     def solveQMRCGSTaB():
#     def solveGPBICG():

# POLYNÔME :
# ----------
# quadratique, cubique, quartique, bissection, sécante, halley, steffesen
# NoorKhanHussain, XiaofengDongweiDongyang, soleymani, jaiswal, wangliu
# chunneta, thukral, HBM1, aberth, WeierstrassDurandKerner, nourein, petkovic, khomosky

# INTÉGRaLE :
# -----------
# trapèze, simpson 1/3, simpson 3/8, romberg, gauss-legendre

# def aireTri(p): a=0
#     for i, _ in enumerate(p):
#     aire += p[i - 1][0] * p[i][1] - p[i][0] * p[i - 1][1]
#     return aire / 2.

# def bezout(a, b):
#     if b == 0:
#         return (1, 0)
#     u, v = bezout(b, a % b)
#         return (v, u - (a // b) * v)

# def get_bit(num, i):
#     return (num & (1 << i)) != 0

# def set_bit(num, i):
#     return num | (1 << i)

# def clear_bit(num, i):
#     mask = ~(1 << i)
#     return num & mask

# def update_bit(num, i, bit):
#     mask = ~(1 << i)
#     return (num & mask) | (bit << i)

# def angle(a,b):
#     n=len(a)
#     m=len(b)
#     if n == m:
#         return np.arccos(np.dot(a, b.T)/(np.linalg.norm(a, ord=np.inf)*np.linalg.norm(b, ord=np.inf)))

#      C
#     /|
#  h / | o
#   /  |
# a --- B
#    a
# sin a = opposé / hypoténuse
# cos a = adnacent / hypoténuse
# tan a = opposé / adnacent

def puissance_de_deux(n):
    return n > 0 and not n & (n-1)

# def inv(a, p):
#     return bezout(a, p)[0] % p

# /**
#  * @brief Function to calculate binomial coefficients
#  * @param n first value
#  * @param k second value
#  * @return binomial coefficient for n and k
#  */
# size_t calculate(int32_t n, int32_t k) {
#     // basic cases
#     if (k > (n / 2))
#         k = n - k;
#     if (k == 1)
#         return n;
#     if (k == 0)
#         return 1;

#     size_t result = 1;
#     for (int32_t i = 1; i <= k; ++i) {
#         result *= n - k + i;
#         result /= i;
#     }

#     return result;
# }

# def binom2(n, k):
#    tableau = np.zeros((n+1,n+1))

#     for i in range(n):
#         t[i] = new long[i+1]
#         t[i][0] = 1
#         for( int j = 1; j<i; ++j)
#             t[i][j] = t[i-1][j] + t[i-1][j-1]
#         t[i][i] = 1
#    return t[n][k]

def factorial(k: int) -> int:
    if k<0:
        return 0
    elif k == 0:
        return 1
    else:
        return k * factorial(k-1)

def binom(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    return binom(n-1, k-1) + binom(n-1,k)
     
def bernstein(n, i, t):
    return (factorial(n)/(factorial(i)*factorial(n-i))) * (1.0-t)**(n-i) * t**i

# def Berstein4(p, t):
#     if p == False:
#         return (1-t)**4
#     elif Point == 1:
#         return 4*t*(1-t)**3
#     elif Point == 2:
#         return 6*t**2*(1-t)**2
#     elif Point == 3:
#         return 4*t**3*(1-t)
#     elif Point == 4:
#         return t**4

# def CourbeBezierQuartiqueRationnelle(Points):

def abcd(a, b, c, d):
    # ax + b = cx + d
    # ax - cx = d-b
    # x(a-c) = d-b
    # x = (a-c)/(d-b)
    x = (d-b) / (a-c)
    return x

def Quadratique(
    a: np.float128,
    b: np.float128,
    c: np.float128
    ) -> np.float128 | tuple[np.float128, np.float128] | tuple[np.complex256, np.complex256]:
    decimales = np.finfo(np.float128).precision #sys.dig
    #np.set_printoptions(suppress=True, precision=deceps)
    tol = sys.float_info.epsilon 
    den = np.float128(2.0*a) # R
    cden = np.complex256(2.0*a) # C
    x: np.float128 = 0.0
    x1: np.float128 = 0.0
    x2: np.float128 = 0.0
    # z1: np.complex256(0.0)
    # z2: np.complex256(0.0)
    z1: complex
    z2: complex
    
    if np.isnan(a) or np.isnan(b) or np.isnan(c): #math.isnan(....)
        raise ValueError("a, B ou C ne sont pas des nombres.")

    if np.isclose(a, 0.0, atol=tol) == True:
        return np.float128(-c/b)
    else:
        if np.isclose(a + b + c, 0.0, atol=tol) == True:
            x = np.float128(c/a)
            disc = np.float128(b**2 - 4.0*a*c)
            if np.isclose(disc, 0.0, atol=tol) == True:
                return x
            else:
                if np.greater(x, 1.0) == True:
                    return 1, np.around(x, decimals=decimales)
                else:
                    return np.around(x, decimals=decimales), 1

        if np.isclose(b, 0.0, atol=tol) == True and np.isclose(c, 0.0, atol=tol) == False:
            disc = np.float128(-4.0*a*c)
            if np.greater(disc, 0.0) == True:
                srdisc = np.float128(np.sqrt(disc))
                x1 = np.float128(-srdisc) / den
                x2 = np.float128(-x1)
                return np.around(x1, decimals=decimales), np.around(x2, decimals=decimales)
            elif np.less(disc, 0.0):
                srdisc = np.emath.sqrt(disc)
                z1 = np.complex256(-srdisc / cden)
                z2 = np.complex256(srdisc / cden)
                return np.around(z1, decimals=decimales), np.around(z2, decimals=decimales)
        else:
            if np.isclose(c, 0.0, atol=tol) == True:
                if np.isclose(a, 1.0, atol=tol):
                    if np.greater(b, 0.0) == True:
                        return -b, 0.0
                    if np.less(b, 0.0) == True:
                        return 0.0, -b
                if np.isclose(a, -1.0, atol=tol):
                    if np.greater(b, 0.0) == True:
                        return 0.0, b
                    if np.less(b, 0.0) == True:
                        return b, 0.0
                if np.greater(b, 0.0) == True:
                    if np.greater(a, 0.0) == True:
                        return np.float128(b/-a), 0.0
                    else:
                        return 0.0, np.float128(b/-a)
                else:
                    if np.greater(a, 0.0) == True:
                        return 0.0, np.float128(b/-a)
                    else:
                        return np.float128(b/-a), 0.0

        disc = np.float128(b*b-4.0*a*c)
        if np.greater(disc, 0.0) == True:
            sd = np.float128(np.sqrt(disc))
            if np.greater(b, 0.0) == True:
                x1 = np.float128((2.0 * c) / ((-b) - sd))
                x2 = np.float128((-b) - sd) / den
            if np.less(b, 0.0) == True:
                x1 = np.float128((-b) + sd) / den
                x2 = np.float128((2.0 * c) / ((-b) + sd))
            return np.around(x1, decimals=decimales), np.around(x2, decimals=decimales)
        elif np.isclose(disc, 0.0, atol=tol) == True:
            x = np.float128(-b) / den
            return np.around(x, decimals=decimales)
        else:
            csd = np.emath.sqrt(disc)
            z1 = np.complex256(-b/cden - csd/cden)
            z2 = np.complex256(-b/cden + csd/cden)
            return np.round(z1, decimals=decimales), np.around(z2, decimals=decimales)


def Cubique(
    a: np.float128,
    b: np.float128,
    c: np.float128,
    d: np.float128
    ) -> \
		np.float128 \
		| tuple[np.float128, np.float128] \
		| tuple[np.float128, np.float128, np.float128] \
		| tuple[np.float128, np.complex256, np.complex256]:
    decimales = np.finfo(np.float128).precision
    tol = sys.float_info.epsilon 
    if np.isclose(a, 0.0, atol=tol) == True:
        return Quadratique(b,c,d)

    disc = np.float128(b**2*c**2 - 4.0*a*c**3 - 4.0*b**3*d - 27.0*a**2*d**2 + 18.0*a*b*c*d)
    p = np.float128((3.0*a*c-b**2) / (3.0*a**2))
    q = np.float128((2.0*b**3+27.0*a**2*d-9.0*a*b*c) / (27.0*a**3))
    t = np.float128(-b / (3.0*a))
    disc_r = np.float128(q**2/4.0 + p**3/27.0)
    sqrdisc = np.float128(np.sqrt(np.fabs(disc_r)))
    omega = np.complex256((-0.5)+np.sqrt(3.0)/2.0j)
    omega2 = np.complex256((-0.5)-np.sqrt(3.0)/2.0j)
    u = np.cbrt((-q/2.0+sqrdisc))
    v = np.cbrt((-q/2.0-sqrdisc))

    if np.greater(disc_r, 0.0) == True:
        x1 = np.float128(u+v+t)
        x2 = omega*np.complex256(u) + omega2*np.complex256(v) + t
        x3 = omega2*np.complex256(u) + omega*np.complex256(v) + t
        return np.around(x1,decimals=decimales), np.around(x2,decimals=decimales), np.around(x3,decimals=decimales)

    # b**2*c**2 - 4.0*a*c**3 - 4.0*b**3*d - 27.0*a**2*d**2 + 18.0*a*b*c*d
    # if np.isclose(disc, 0.0, atol=tol) == True:

    # # 1 racine triple
    # if np.isclose(b**2*c**2 - 4.0*a*c**3 - 4.0*b**3*d - 27.0*a**2*d**2, 18.0*a*b*c*d, atol=tol) == True:

    # # 1 racine double < 1 racine simple
    # if np.isclose(b**2*c**2 - 4.0*a*c**3 - 4.0*b**3*d + 18.0*a*b*c*d = 27.0*a**2*d**2, atol=tol) == True:

    # #  1 racine simple < 1 racine double
    # if np.isclose(b**2*c**2 - 4.0*a*c**3 - 4.0*b**3*d + 18.0*a*b*c*d = -27.0*a**2*d**2, atol=tol) == True:

    if np.isclose(disc_r, 0.0, atol=tol):
        x1 = np.float128((-1.0)*((fabs((-4.0)*-q))**(1.0/3.0)-t))
        x2 = np.float128((-1.0)*(-(fabs(-q/2.0))**(1.0/3.0)-t))
        if np.isclose(x1, x2, atol=tol) == True:
            return np.around(x1,decimals=decimales)
        else:
            return np.around(x1,decimals=decimales), np.around(x2.real,decimals=decimales)

    if np.less(disc_r, 0.0):
        theta = np.arccos(-np.sqrt((-27*q**2) / (4*p**3)))
        x1 = np.float128(2.0 * np.sqrt((-p/3.0) * np.cos(theta/3))) + t
        x2 = np.float128(2.0 * np.sqrt((-p/3.0) * np.cos((theta + 2.0*np.pi)/3))) + t
        x3 = np.float128(2.0 * np.sqrt((-p/3.0) * np.cos((theta + 4.0*np.pi)/3))) + t
        return np.around(x1.real,decimals=decimales), np.around(x2.real,decimals=decimales), np.around(x3.real,decimals=decimales)

def Cubique2(
    a: np.float128,
    b: np.float128,
    c: np.float128,
    d: np.float128
    ) -> \
		np.float128 \
		| tuple[np.float128, np.float128] \
		| tuple[np.float128, np.float128, np.float128] \
		| tuple[np.float128, np.complex256, np.complex256]:

    decimales = np.finfo(np.float32).precision
    tol = sys.float_info.epsilon
    if np.isclose(a, 0.0, atol=tol) == True:
        raise ValueError("Ce n'est pas une équation cubique car a=0.")
        return Quadratique(b,c,d)

    mm.dps = 7
    mm.pretty = False

    disc = b**2*c**2 - 4*a*c**3 - 4*b**3*d - 27*a**2*d**2 + 18*a*b*c*d
    p = mm.fdiv(3*a*c - b**2, 3*a**2)
    q = mm.fdiv(2*b**3 + 27*a**2*d - 9*a*b*c, 27*a**3)
    t = mm.fdiv(-b, 3*a)
    disc_r = mm.fdiv(q*q, 4) + mm.fdiv(p*p*p, 27)

    sqrdisc = mm.sqrt(mm.fabs(disc_r))

    omega = mm.mpc(-0.5,mm.sqrt(3)/2)
    omega2 = mm.conj(omega)

    u = mm.cbrt((-q/2 + sqrdisc))
    v = mm.cbrt((-q/2 - sqrdisc))

    if disc_r > tol:
        print("Delta > 0")
        x1 = u + v + t
        x2 = omega*mm.mpc(u) + omega2*mm.mpc(v) + t
        x3 = omega2*mm.mpc(u) + omega*mm.mpc(v) + t
        return x1, x2, x3

    if disc_r >= -tol or disc_r <= tol:
        print("Delta = 0")
        x1 = mm.power(mm.fabs((-4) * q), 1/3) - t
        x2 = (-1) * mm.power(mm.fabs(-q / 2), 1/3) - t
        if np.isclose(x1, x2, atol=tol) == True:
            return x1
        else:
            return x1, x2

    if disc_r < tol:
        print("Delta < 0")
        theta = mm.acos(-mm.sqrt(mm.fdiv(-27*q*q, 4*p*p*p)))
        x1 = 2 * mm.sqrt(mm.fdiv(-p,3) * mm.cos(theta/3)) + t
        x2 = 2 * mm.sqrt(mm.fdiv(-p,3) * mm.cos((theta + 2 * mm.pi)/3)) + t
        x3 = 2 * mm.sqrt(mm.fdiv(-p,3) * mm.cos((theta + 4 * mm.pi)/3)) + t
        return np.real(x1), np.real(x2), np.real(x3)

def Quartic(a,b,c,d,e):
    if a == 0.0 and b == 0.0:
        return Quadratique(c,d,e)
    elif a == 0.0:
        return Cubique2(b,c,d,e)
    else:
        p = (8.0*a*c - 3.0*b**2) / (8.0*a**2)
        q = (b**3 - a*b*c + 8.0*a**2*d) / (8.0*a**3)
        r = (-3.0*b**4 + 16.0*a*b**2*c - 64.0*a**2*b*d + 256.0*a**3*e) / (256.0*a**4)
        disc = 16.0*p**4*r - 4.0*p**3*q**2 - 128.0*p**2*r**2 + 144.0*p*q**2*r - 27.0*q**4 + 256.0*r**3

        if disc == np.float128(0.0):
            t = b/(4.0*a)
            s = mm.sqrt(mm.fabs(p**2-4.0*r))
            a = -p
            add = a+s
            sub = a-s
            x1 = mm.sqrt(add/2.0)-t
            x2 = mm.sqrt(sub/2.0)-t
            x3 = -mm.sqrt(add/2.0)-t
            x4 = -mm.sqrt(sub/2.0)-t
            return np.around(x1,decimals=8), np.around(x2,decimals=8), np.around(x3,decimals=8), np.around(x4,decimals=8)
        else:
            t = b/(4.0*a)
            y1, y2, y3 = Cubique(1.0,2.0*p,p**2-4.0*r,-q**2)
            u = mm.sqrt(-y1)
            v = mm.sqrt(-y2)
            w = mm.sqrt(-y3)
            x1 = (1.0/2.0) * (u+v+w) - t
            x2 = (1.0/2.0) * (u-v-w) - t
            x3 = (1.0/2.0) * (-u+v-w) - t
            x4 = (1.0/2.0) * (-u-v+w) - t
            return np.around(x1,decimals=8),np.around(x2,decimals=8),np.around(x3,decimals=8),np.around(x4,decimals=8)

def QuarticFerrari(a,b,c,d,e):
    c1 = b/a
    c2 = c/a
    c3 = d/a
    c4 = e/a

    return Cubique2(1, -c2**2/2.0, (1.0/4.0)*(c1*c3-4.0*c4), (1.0/8.0)*(4.0*c2*c4-c1**2*c4-c3**2))

def QuarticDescartes(a,b,c,d,e):
    if a == 0.0 and b == 0.0:
        return Quadratique(c,d,e)
    elif a == 0.0:
        return Cubique2(b,c,d,e)
    else:
        p = (8.0*a*c - 3*b**2) / (8*a**2)
        q = (b**3 - a*b*c + 8*a**2*d) / (8*a**3)
        r = (-3*b**4 + 16*a*b**2*c - 64*a**2*b*d + 256*a**3*e) / (256*a**4)
        disc = 16*p**4*r - 4*p**3*q**2 - 128*p**2*r**2 + 144*p*q**2*r - 27*q**4 + 256*r**3

        if disc == np.float128(0.0):
            t = b/(4*a)
            s = np.sqrt(np.fabs(p**2-4*r))
            a = -p
            add = a+s
            sub = a-s
            x1 = np.sqrt(add/2)-t
            x2 = np.sqrt(sub/2)-t
            x3 = -np.sqrt(add/2)-t
            x4 = -np.sqrt(sub/2)-t
            return np.around(x1,decimals=8), np.around(x2,decimals=8), np.around(x3,decimals=8), np.around(x4,decimals=8)
        else:
            return 0

# Solving cubics by polynomial fitting : à finir
def cubic_strobach(a,b,c,d,m):
    qq = (a*a-3.0*b) / 9.0
    rr = (2.0*a*a*a-9.0*a*b+27.0*c) / 54.0
    qqq = qq*qq*qq
    rrr = rr*rr
    x3hat = 0.0
    if rrr < qqq:
        theta = np.arccos(rr/qq**1.5)
        x3hat = -2.0*np.sqrt(qq) * np.cos(theta/3.0) - a/3.0
    else:
        aa = -np.sign(rr) * (np.fabs(rr) + np.sqrt(rrr-qqq))**(1.0/3.0)
        if np.isclose(a, 0.0, atol=1e-16) == False:
            bb = qq/aa
        else:
            bb = 0
        x3hat = aa + bb - a/3.0
    eee = 0.0
    ee = 0.0
    gamma = -x3hat
    alpha = a - gamma
    beta = b - gamma*alpha
    e1 = 0
    e2 = 0
    e3 = c-gamma*beta
    for mu in range(16):
        eeee=eee
        eee=ee
        u1 = alpha - gamma
        u2 = beta - gamma*u1
        q1 = e1
        q2 = e2 - gamma*q1
        q3 = e3 - gamma*q2
        if np.isclose(u2, 0.0, atol=1e-16) == True:
            delta3 = 0.0
        else:
            delta3 = q3/u2
        delta2 = q2 - u1*delta3
        delta1 = q1 - delta3
        alpha += delta1
        beta += delta2
        gamma += delta3
        e1 = a - gamma - alpha
        e2 = b - alpha*gamma - beta
        e3 = c - gamma*beta
        ee = e1*e1 + e2*e2 + e3*e3
        if ee == 0.0 or ee == eee or  ee == eeee:
            break


def derivative1(
    f: Callable,
    x: np.float128,
    h: np.float128 = 1e-15
    ) -> np.float128:
    return (f(x-2.0*h)-8.0*f(x-h)-8.0*f(x+h)-f(x-2.0*h))/(12.0*h)

def derivative2(
    f: Callable,
    x: np.float128,
    h: np.float128 = 1e-15
    ) -> np.float128:
    return (-f(x-2.0*h)+16.0*f(x-h)-30.0*f(x+h)+16.0*f(x+h)-f(x-2.0*h))/(12.0*h*h)

def derivative3(
    f: Callable,
    x: np.float128,
    h: np.float128 = 1e-15
    ) -> np.float128:
    return (f(x-3.0*h)-8.0*f(x-2.0*h)+13.0*f(x-h)-13.0*f(x+h)+8.0*f(x+2.0*h)-f(x+3.0*h))/(8.0*h*h*h)

def Taux(x_approx: np.float128, x_exact: np.float128) -> np.float128:
    e = [np.abs(x_inc, x_exact) for x_inc in x_approx]
    c = [np.log(e[n+1]/e[n]) / np.log(e[n]/e[n-1]) for n in range(1,len(e)-1,1)]
    return c

def Dichotomie(
    f: Callable,
    a:np.float128,
    b:np.float128,
    k:int,
    tol: np.float128 = 1e-15
    ) -> np.float128:

    fa = f(a)
    if np.isclose(fa, 0.0, atol=tol) == True:
        return a
    fb = f(b)
    if np.isclose(fb, 0.0, atol=tol) == True:
        return b
    if np.greater(fa*fb, 0.0) == True:
        print("L'interval choisi [{},{}] est incorrect.".format(a,b))
        exit()
    co:np.float128 = 0
    print(repr('a').rjust(12),
            repr('b').rjust(12),
            repr('cn').rjust(12))
    print("-" * 38)
    for _ in range(k):
        cn = a + 0.5 * (b-a)
        fc = f(cn)
        if np.isclose(fc, 0.0, atol=tol) or np.fabs(cn-co) < tol:
            break
        else:
            if np.less(fa*fc, 0.0):
                b = cn
                fb = fc
            elif np.greater(fa*fc, 0.0):
                a = cn
                fc = f(cn)
        print(repr(np.around(a, decimals=8)).rjust(12),
            repr(np.around(b, decimals=8)).rjust(12),
            repr(np.around(cn, decimals=8)).rjust(12))
        # print(f"{a:12.8f}{b:12.8f}{cn:12.8f}")
        co = cn

    return cn

# https://www.arcjournals.org/pdfs/ijsimr/v7-i9/3.pdf
# http://www.m-hikari.com/ams/ams-2020/ams-5-8-2020/p/junaMS5-8-2020.pdf
def Quadrisection(
    f: Callable,
    a: np.float128,
    b: np.float128,
    k: int,
    tol: np.float128 = 1e-15) -> np.float128:
    fa = f(a)
    if (fa == 0):
        return a
    fb = f(b)
    if (fb == 0):
        return b
    for _ in range(k):
        h = (b-a)/4.0
        c1 = a+h
        c2 = a+2.0*h
        c3 = a+3.0*h
        p = (c1+c2+c3)/3.0
        print("fabs(b-p)=",np.fabs(b-p))
        if np.fabs(b-p) <= tol:
            break
        else:
            if f(a)*f(c1) < 0.0:
                b = c1
            elif f(c1)*f(c2) < 0.0:
                a = c1
                b = c2
            elif f(c2)*f(c3) < 0.0:
                a = c2
                b = c3
            else:
                a = c3

def Pegasus(f, a, b, k, err) -> np.float128:
    return 0

def Secante(
    f: Callable,
    x0: np.float128,
    x1: np.float128,
    k: int,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    fx0 = f(x0)
    fx1 = f(x1)
    for _ in range(k):
        try:
            d = (fx0-fx1)
            x = x1 - fx1*(x0-x1) / d
        except ZeroDivisionError:
            print("Division par zéro d = ", x)
            sys.exit(1)
        x0 = x1
        fx0 = fx1
        x1 = x
        fx1 = f(x1)
        if np.less(np.fabs(x1-x0), tol) == True:
            break

    return x1

def RegulaFalsi(
    f: Callable,
    x0: np.float128,
    x1: np.float128,
    k: int,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    fx0 = f(x0)
    fx1 = f(x1)
    x2: np.float128 = 0.0
    for _ in range(k):
        x2 = (x0*fx1-x1*fx0) / (fx1-fx0)
        fx2 = f(x2)
        if np.less(np.fabs(fx2), tol) == True:
            break
        if np.less(fx2*fx0, 0.0) == True:
            x1 = x2
            fx1 = fx2
        else:
            x0 = x2
            fx0 = fx2

    return x1

def NewtonRaphson(
    f: callable,
    df: callable,
    x0: np.float128,
    k: int,
    # retlist = False,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec: np.float128 = np.finfo(np.float128).precision
    xo: np.float128 = x0
    xn: np.float128 = xo

    if f == None or df == None:
        raise RuntimeError("aucune fonction n'a été passée en argument.")
    if tol <= 0.0:
        raise ValueError("L'epsilon doit être strictement positif.")

    # if retlist:
    #     it = []

    for _ in range(k):
        fxn: np.float128 = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn: np.float128 = df(xn)
        try:
            xn = xo-fxn / dfxn
        except ZeroDivisionError:
            print("df(x) = {:g}".format(dfxn))
            sys.exit(1)
        if np.less(np.fabs(xn-xo)/np.fabs(xo), tol) == True or k == 0:
            break
        xo = xn
        # if retlist:
        #     it.append(xn)
    else:
        raise ValueError(f"aucune solution convenable n'a été trouvée durant ces {k} itérations.")

    # if retlist:
    #     return k, it, np.format_float_positional(np.float128(xn), \
    #         unique=False, precision=dec)
    # else:
    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# def NewtonRaphson(
    # f: Callable,
    # df: Callable,
#     x0: np.float128,
#     k: int,
#     tol: np.float128 = 1e-15
#     ) -> np.float128:
#     dec = np.finfo(np.float128).precision
#     xo = x0
#     xn = xo
#     xt = xo
#     coc: np.float128 = 0.0
#     if f == None or df == None:
#         raise RuntimeError("aucune fonction n'a été passée en argument.")
#     if tol <= 0.0:
#         raise ValueError("L'epsilon doit être strictement positif.")
#     for n in range(k):
#         fxn = f(xn)
#         if np.isclose(fxn, 0.0, atol=tol) == True:
#             break
#         dfxn = df(xn)
#         try:
#             xn = xo-fxn / dfxn
#         except ZeroDivisionError:
#             print("df(x) = 0 !")
#             sys.exit(1)
#         if np.less(np.fabs(xn-xo), tol) == True:
#             break
#         xo = xn
#         xt = xo
#         # print(traceback.extract_stack())
#         # if n > 1:
#         #     coc = np.log(np.abs((xn-alpha)/(xo-alpha))) / np.log(np.abs((xn-alpha)/(xt-alpha)))
#     else:
#         raise ValueError(f"aucune solution convenable n'a été trouvée durant ces {k} itérations.")

#     return np.format_float_positional(np.float128(xn), \
#         unique=False, precision=dec)

def Halley(
    f: callable,
    df: callable,
    df2: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break;
        dfxn = df(xn)
        d2fxn = df2(xn)
        d = (2*dfxn**2 - fxn*d2fxn)
        if (np.isclose(d, 0.0, atol=tol) == False):
            xn = xo - 2.0*(fxn*dfxn) / d
            if np.less(np.fabs(xn-xo), tol) == True:
                break
            xo = xn
    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

def Steffesen(
    f: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        xnd = (f(xn+fxn) - fxn)
        if (np.isclose(xnd, 0.0, atol=tol) == False):
            xn = xo - fxn**2 / xnd
            if np.less(np.fabs(xn-xo), tol) == True:
                break
            xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Fourth-order iterative method without calculating the higher derivatives for nonlinear equation
def Li(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
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
        xn = xn - (fxn * (fxn - fyn)) / (dfxn * (fxn - 2.0*fyn))
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# New Modification of Chebyshev's Method with Seventh-Order Convergence
def MuhaijirSolehSafitri(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
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
        n1 = 2.0 * ((fyn-fxn) / (yn-xn)) - dfxn
        n2 = 2.0 * (((fyn-fxn)-(yn-xn)*dfxn)/((xn-yn)*(xn-yn)))
        zn = yn - (fyn / n1) * (1.0 + 0.5 * (fyn*n2)/(n1*n1))
        fzn = f(zn)
        fxz = np.float128((fzn-fxn) / (zn-xn))
        fyz = np.float128((fzn-fyn) / (zn-yn))
        fxy = np.float128((fyn-fxn) / (yn-xn))
        xn = zn - fzn / (fxz + fyz - fxy)
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# a new modified Halley method without second derivatives for nonlinear equation
def NoorKhanHussain(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break;
        dfx = df(xn)
        yn = xn - fxn/dfx
        fyn = f(yn)
        dfy = df(yn)
        d1 = 2*fxn * dfx**2
        d2 = dfx**2 * fyn + dfx*dfy*fyn
        xn = np.float128(yn - (2.0 * fxn * fyn * dfy) \
            / (d1-d2))
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# a Modified Super-Halley's Method Free From Second Derivative
def XiaofengDongweiDongyang(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
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
        zn = np.float128(xn-(1.0+(fyn/(fxn-2.0*fyn))) \
            * fxn/dfxn)
        fzn = f(zn)
        xn = np.float128(zn + fzn/(dfxn \
            + ((2.0*fyn*dfxn**2)/(fxn**2))*(zn-xn)))
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Steffensen type methods for solving nonlinear equations
def CorderoHuesoMartinezTorregrosa(
    f: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        yn = xo - (2.0 * fxn*fxn) / (f(xn+fxn) - f(xn-fxn))
        fyn = f(yn)
        zn = yn - ((yn-xn) / (2.0*fyn-fxn)) * fyn
        fzn = f(zn)
        xn = zn - ((yn-xn) / (2.0*fyn-fxn)) * fzn
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Optimal Derivative-Free Root Finding Methods Based on Inverse Interpolation
def JunjuaZafarYasmin_MNP16(
    f: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = np.float128(f(xn))
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        zn = np.float128(xn + fxn**4)
        fzn = np.float128(f(zn))
        fzx = np.float128((fxn-fzn)/(xn-zn))
        yn = np.float128(xo - fxn/fzx)
        fyn = f(yn)
        fyx = np.float128((fxn-fyn)/(xn-yn))
        fyn_fxn = fyn-fxn
        h2 = 1.0 / ((fyn_fxn)*fyx) - 1.0 / ((fyn_fxn)*fzx)
        wn = yn + h2*fxn**2
        fwn = f(wn)
        fwx = np.float128((fxn-fwn)/(xn-wn))
        fyn_fwn = fyn-fwn
        fwn_fxn = fwn-fxn
        g3_1 = np.float128(1.0 / ((fyn_fxn)*(fyn_fwn)*fyx))
        g3_2 = np.float128(1.0 / ((fwn_fxn)*(fyn_fwn)*fwx))
        g3_3 = np.float128(1.0 / ((fwn_fxn)*(fyn_fwn)*fzx))
        g3_4 = np.float128(1.0 / ((fyn_fxn)*(fyn_fwn)*fzx))
        g3 = g3_1 - g3_2 + g3_3 - g3_4
        b3 = 1.0 / ((fyn_fxn)*fyx) - 1.0 / (fzx*(fyn_fxn)) - g3*(fyn_fxn)
        tn = yn + b3*fxn**2 - g3*fxn**3
        ftn = f(tn)
        ftx = np.float128((fxn-ftn)/(xn-tn))
        ftn_fxn = ftn-fxn
        phi_t = np.float128(1.0 / (ftx*(ftn_fxn)) - 1.0 / (fzx*(ftn_fxn)))
        phi_w = np.float128(1.0 / (fwx*(fwn_fxn)) - 1.0 / (fzx*(fwn_fxn)))
        phi_y = np.float128(1.0 / (fyx*(fyn_fxn)) - 1.0 / (fzx*(fyn_fxn)))
        g5 = np.float128(((phi_t-phi_w)/(ftn-fwn) - (phi_y-phi_w)/(fyn_fwn)) / (ftn-fyn))
        g4 = np.float128((phi_t-phi_w)/(ftn-fwn) - g5*((ftn_fxn)+(fwn_fxn)))
        b4 = np.float128(phi_t-g4*(ftn_fxn) - g5*(ftn_fxn)*(ftn_fxn)**2)
        xn = yn + b4*fxn**2 - g4*fxn**3 + g5*fxn**4
        print("xn = ", xn)
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# a sixth-order derivative-free iterative method for solving nonlinear equations
def EdwarImranDeswita(
    f: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        t1x = (f(xn+fxn) - f(xn-fxn)) / (2.0*fxn)
        yn = xo - fxn/t1x
        fyn = f(yn)
        t1y = (f(yn+fyn) - f(yn-fyn)) / (2.0*fyn)
        zn = yn - (fxn*(t1x-t1y)) / (2.0*t1x*t1x)
        fzn = f(zn)
        xn = zn - (2.0*fzn*t1x) / (4.0*t1x*t1y - (t1x*t1x) - (t1y*t1y))
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Two New Iterative Methods for Solving Nonlinear Equations without Derivative
def KantaloMuangchanSompong(
    f: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        yn = xo - (4.0*fxn*fxn) / (3.0*f(xn+fxn) - 2.0*fxn-f(xn-fxn))
        fyn = f(yn)
        xn = xn - (4.0*fxn*fxn) / (3.0*f(xn+fxn) - 2.0*fxn-f(xn-fxn)*(fxn-fyn))
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# _equ27 a New Optimal Eighth-Order Ostrowski-Type Family of Iterative Methods for Solving Nonlinear Equations
def LotfiEftekhari_equ_27(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
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
        zn = yn - fxn/(fxn-2.0*fyn) * (fyn/dfxn)
        fzn = f(zn)
        fxy = np.float128((fxn-fyn) / (xn-yn))
        fxz = np.float128((fxn-fzn) / (xn-zn))
        fyz = np.float128((fyn-fzn) / (yn-zn))
        xn1 = (1.0 + np.sin(fzn/fxn))
        xn2 = (1.0 + (fyn**4/fxn**4) * np.cos(fyn/fxn))
        xn3 = np.cos(fzn/fyn) * ((fzn*fxy) / (fxz*fyz))
        xn = zn - xn1*xn2*xn3
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Optimal Eighth Order Convergent Iteration Scheme Based on Lagrange Interpolation
def SharmaBahl(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k:int,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = np.float128(f(xn))
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn = np.float128(df(xn))
        wn = np.float128(xo - fxn / dfxn)
        fwn = np.float128(f(wn))
        zn = np.float128(wn - fxn/(fxn - 2*fwn) * fwn/dfxn)
        fzn = np.float128(f(zn))
        fwz = np.float128((fzn-fwn)/(zn-wn))
        fxw = np.float128((fwn-fxn)/(wn-xn))
        fxz = np.float128((fzn-fxn)/(zn-xn))
        xn1 = np.float128(((fxn**3)/(fxn**3 + fwn**3) + fzn/fxn + fzn**2/fxn**2))
        xn2 = fzn / (fxz+fwz-fxw)
        xn = zn - xn1 * xn2
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Regarding the accuracy of optimal eighth-order methods
def Soleymani(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn = df(xn)
        yn = xo - (2.0/3.0) * (fxn/dfxn)
        dfyn = df(yn)
        zn = xn - (1.0-(3.0/8.0) * (dfyn*dfyn-dfxn*dfxn)/(dfyn*dfyn)) * (fxn/dfxn)
        fzn = f(zn)
        fzx = (fxn-fzn)/(xn-zn)
        fzxx = 2.0 * (fzx-dfxn)/(xn-zn)
        xn = zn - (fzn)/(dfyn + 2.0*fzxx * (zn-yn))
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Optimized Steffensen-Type Methods with Eighth-Order Convergence and High Efficiency Index
def Soleymani2(
    f: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        wn = xn - fxn
        fwn = f(wn)
        fxw = (fwn-fxn)/(wn-xn)
        yn = xo - (fxn/fxw)
        fyn = f(yn)
        zn1 = (1.0+(fyn/fxn)+(fyn/fxn)**2)
        zn2 = (1.0+(fyn/fwn)+(3.0-2.0*fxw*(fyn/fwn)**2))
        zn = yn - (fyn/fxw) * (zn1*zn2)
        fzn = f(zn)
        w2_1 = np.float128(1.0+(fzn/fyn)+(fzn/fyn)**2 \
                +(fyn/fwn)+(5.0-4.0*fxw)*(fyn/fwn)**2)
        w2_2 = np.float128((10.0+2.0*fxw*(-8.0+3.0*fxw)) \
                *(fyn/fwn)**3)
        w2_3 = np.float128((11.0-fxw*(26.0-15.0*fxw+fxw**3)) \
                *(fyn/fwn)**4)
        w2_4 = np.float128((fyn/fxw)+(fyn/fxw)**2 \
                +(4.0-2.0*fxw)*(fzn/fwn))
        w2 = w2_1+w2_2+w2_3+w2_4
        xn = zn - (fzn/fxw) * w2
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Optimal Fourth, Eighth and Sixteenth Order Methods by Using Divided Difference Techniques and Their Basins of attraction and Its application
def TaoMadhu(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k: int = 10,
    tol = np.finfo(np.float64).eps
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn = df(xn)
        psi2 = xo - fxn/dfxn
        fpsi2 = f(psi2)
        fpsi2xx = (fpsi2-dfxn) / (fpsi2-xn)
        psi4 = psi2 - fpsi2 / (dfxn + (2.0*fpsi2xx)*(psi2-xn))
        fpsi4 = f(psi4)
        fpsi4xx = (fpsi4-dfxn) / (fpsi4-xn)
        b2 = np.float128((fpsi2xx*(psi4-xn) - fpsi4xx*(psi2-xn)) / (psi4-psi2))
        b3 = np.float128((fpsi4xx - fpsi2xx) / (psi4-psi2))
        psi8 = psi4 - fpsi4 / (dfxn + 2.0*b2*(psi4-xn) + 3.0*b3* np.power((psi4-xn),2))
        fpsi8 = f(psi8)
        fpsi8xx = (fpsi8-dfxn) / (fpsi8-xn)
        s1 = (psi2-xn)
        s2 = (psi4-xn)
        s3 = (psi8-xn)
        s1s1 = s1*s1
        s2s2 = s2*s2
        s3s3 = s3*s3
        d = (-s1s1*s2+s1*s2s2+s1s1*s3-s2s2*s3-s1*s3s3+s2*s3s3)
        c2_1 = (fpsi2xx * (-s2s2*s3+s2*s3s3) + fpsi4xx * (s1s1*s3-s1*s3s3) + fpsi8xx * (-s1s1*s2-s1*s2s2))
        c2_2 = d
        c2 = c2_1 / c2_2
        c3_1 = (fpsi2xx * (s2s2-s3s3) + fpsi4xx * (-s1s1+s3s3) + fpsi8xx * (s1s1-s2s2))
        c3_2 = d
        c3 = c3_1 / c3_2
        c4_1 = (fpsi2xx * (-s2+s3) + fpsi4xx * (s1-s3) + fpsi8xx * (-s1+s2))
        c4_2 = d
        c4 = c4_1 / c4_2
        psi16 = psi8 - fpsi8 / (dfxn + 2.0*c2*s3 + 3.0*c3*s3s3 + 4.0*c4*s3s3*s3)
        xn = psi16
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# a new optimal family of three-step methods for efficient finding of a simple root of a nonlinear equation
def RalevicCebic_alg1(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn = df(xn)
        wn = xo - fxn/dfxn
        fwn = f(wn)
        fwx = np.float128(((fxn-fwn)/(xn-wn)))
        zn = wn - (3.0 - (2.0*fwx)/dfxn) * (fwn/dfxn)
        fzn = f(zn)
        fzx = ((fxn-fzn)/(xn-zn))
        fzw = ((fwn-fzn)/(wn-zn))
        xn = zn + (fzn/fzx) * ((fzw)/(fzx-2.0*fzw))
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# New efficient Steffensen Type Method for Solving Nonlinear Equations
def Jaiswal(
    f: Callable,
    x0: np.float128,
    k:int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        yn = xo - (2*fxn**2) / (f(xo+fxn)-f(xo-fxn))
        fyn = f(yn)
        xn1 = np.float128((2.0*fxn*(fxn+fyn)) \
            /(f(xo+fxn)-f(xo-fxn)))
        xn2 = np.float128((4.0*fxn**3) \
            / ((f(xo+fxn)-f(xo-fxn))*(fxn-fyn)))
        xn = xo + xn1-xn2
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# New three step derivative free iterative method for solving nonlinear equations
def tdsm(
    f: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xo)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        an = fxn**2 / (f(xn+fxn) - fxn)
        fan = f(an)
        bn = xn - fxn / (2*(fan-fxn)/(an-xn) - fxn/(an-xn))
        fbn = f(bn)
        xn = xn - (fxn*(bn-xn)) / (fbn-fxn)
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Optimal fourth- and eighth- order of convergence derivative-free modifications of King’s method
def SolaimanHashim_equ17(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn = np.float128(df(xn))
        yn = np.float128(xo - fxn/dfxn)
        fyn = np.float128(f(yn))
        dfyn = np.float128(df(yn))
        qxy = np.float128(2/(xn-yn) * (3*(fxn-fyn)/(xn-yn) - 2*dfyn-dfxn))
        xn = np.float128(yn - fyn/dfyn - (2*fyn**2 * dfyn * qxy) / (4*dfyn**4 - 4*fyn*dfyn**2*qxy + fyn**2*qxy**2))
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Sixteenth-Order Iterative Method for Solving Nonlinear Equations
def JanngamaComemuangEqu23(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k:int = 10,
    tol: np.float128 = 1e-15
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
        dfyn = df(yn)
        d2fyn = 2.0/(yn-xn) * (2.0*dfyn+dfxn-3.0*(fyn-fxn)/(yn-xn))
        pn = (fyn*d2fyn)/dfyn**2
        zn = (2.0*fyn) / (dfyn*(1.0+np.sqrt(1.0-2.0*pn)))
        fzn = f(zn)
        dfzn = df(zn)
        dn = 2.0/(zn-yn) * (2.0*dfzn+dfyn-3.0*(fzn-fyn)/(zn-yn))
        xn = zn - fzn/dfzn - (fzn*dn)/(2.0*dfzn**3)
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# New Eighth and Sixteenth Order Iterative Methods to Solve Nonlinear Equations
def RafiullahJabeen(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k:int = 10,
    tol: np.float128 = 1e-15
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
        dfyn = df(yn)
        zn = yn - fyn/dfyn - ((fyn*fyn) * (dfxn-dfyn))/(2.0*(fxn-fyn)*(dfxn*dfxn))
        fzn = f(zn)
        vn1 = fzn*((xn-yn)*(xn-zn)*(yn-zn))
        vn2 = (-fzn*(xn-yn)*(xn-2.0*zn+yn)+fyn*(xn-zn)*(xn-zn)-fxn*(yn-zn)*(yn-zn))
        vn = zn - vn1/vn2
        fvn = f(vn)
        dfvn1 = fvn/(vn-xn) + fvn/(vn-yn) + fvn/(vn-zn)
        dfvn2 = (fxn*(vn-yn)*(vn-zn)) / ((xn-vn)*(xn-yn)*(xn-zn))
        dfvn3 = (fyn*(vn-xn)*(vn-zn)) / ((vn-yn)*(xn-yn)*(yn-zn))
        dfvn4 = (fzn*(vn-xn)*(vn-yn)) / ((zn-vn)*(zn-xn)*(zn-yn))
        dfvn = dfvn1 + dfvn2 + dfvn3 + dfvn4
        xn = vn - fvn/dfvn
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), unique=False, precision=dec)

# a Family of Optimal Derivative Free Iterative Methods with Eighth-Order Convergence for Solving Nonlinear Equations
def Matinfaraminzadeh_eq21(
    f: Callable,
    x0: np.float128,
    alpha: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        yn = xo - fxn**2 / (f(xn+fxn)-fxn)
        fyn = f(yn)
        zn1 = 0.5*alpha*(fyn*fxn)**2 + 1.5*(fyn/fxn) + 1.0
        zn2 = fxn*fyn / (f(xn+fxn)-fxn)
        zn = yn - zn1*zn2
        fzn = f(zn)
        fkx = 1.0
        psi_1 = ((fzn-yn+zn)*(fzn+zn-xn)*(fxn-fzn-zn+xn)*fzn) / ((xn-zn)*(yn-zn)*(-zn+fxn+xn))
        psi_2 = (fzn*(fzn+zn-xn)*(fxn-fzn-zn+xn)*fyn)/((xn-yn)*(yn-zn)*(xn-yn+fxn))
        psi_3 = (fzn*(fzn+zn-xn)*(fzn-yn+zn)*fkx) / (fxn*(xn-yn+fxn)*(-zn+fxn+xn))
        psi_4 = (fzn*(fzn-yn+zn)*(fxn-fzn-zn+xn)) / ((xn-yn)*(xn-zn))
        psi = psi_1-psi_2+psi_3+psi_4
        xn = zn - fzn**2 / (psi-fzn)
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), unique=False, precision=dec)

# Higher-Order Derivative-Free Iterative Methods for Solving Nonlinear Equations and Their Basins of attraction
def LiWangMadhu_PM16(
    f: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        wn = xn + np.power(fxn,3)
        fwn = f(wn)
        yn = xo - fxn**4 / (fwn-fxn)
        fyn = f(yn)
        fyw = np.float128((fyn-fwn) / (yn-wn))
        fwx = np.float128((fwn-fxn) / (wn-xn))
        fywx = np.float128((fyw-fwx) / (yn - xn))
        zn = yn - fyn / (fyw + (yn-wn) * fywx)
        fzn = f(zn)
        fzy = np.float128((fzn-fyn) / (zn-yn))
        fzyw = np.float128((fyw-fzy) / (wn-zn))
        fzywx = np.float128((fzyw-fywx) / (xn-zn))
        pn_1 = fzy + (zn-yn) * fzyw
        pn_2 = (zn-yn) * (zn-wn) * fzywx
        pn = fzn / (pn_1 + pn_2)
        fpn = f(pn)
        fpz = (fpn-fzn) / (pn-zn)
        fpzy = (fzy-fpz) / (yn-pn)
        fpzyw = (fpzy-fzyw) / (pn-wn)
        fpzywx = (fpzyw-fzywx) / (pn-xn)
        n4_1 = fpz + (pn-zn) * fpzy
        n4_2 = (pn-zn) * (pn-yn) * fpzyw
        n4_3 = (pn-zn) * (pn-yn) * (pn-wn) * fpzywx
        n4 = n4_1 + n4_2 + n4_3
        xn = pn - fpn / n4
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# a new family of optimal eighth order methods with dynamics for nonlinear equations
# # Janak Raj Sharma, Himani arora

# Modified Ostrowski's method with eighth-order convergence and high efficiency index
def WangLiu(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k:int = 10,
    tol: np.float128 = 1e-15
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
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Two new efficient sixth order iterative methods for solving nonlinear equations
def SolaimanHashim(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k:int = 10,
    tol: np.float128 = 1e-15
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
        dfyn = df(yn)
        qxy = (2.0 / (xn-yn)) * (3.0 * ((fxn-fyn)/(xn-yn)) - 2.0*dfyn-dfxn)
        xn1 = 2.0*fyn*fyn * dfyn * qxy
        xn2 = 4.0* np.power(dfyn,4) - 4.0*fyn*dfyn*dfyn*qxy + fyn*fyn*qxy*qxy
        xn = yn - fyn/dfyn - xn1/xn2
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Optimal Eighth-Order Solver for Polynome Equations with applications in Chemical Engineering
def SolaimanHashim2(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k:int = 10,
    tol: np.float128 = 1e-15
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
        dfyn = df(yn)
        rxy = np.float128(((3.0)*(fyn-fxn)/(yn-xn)-2.0*dfyn-dfxn) * (2.0/(xn-yn)))
        fyx = (fxn-fyn)/(xn-yn)
        qyn = 2.0*fyx-dfxn
        wn = yn-fyn/qyn - (2.0*fxn**2*rxy)/(4.0* np.power(qyn,4)-4.0*fyn*qyn**2*rxy+fyn**2*rxy**2)
        fwn = f(wn)
        fxy = (fyn-fxn)/(yn-xn)
        fwx = (fxn-fwn)/(xn-wn)
        kwn = np.float128(fwx*(2.0+(xn-wn)/(yn-wn)) - (xn-wn)**2 /((xn-yn)*(yn-wn))*fxy + dfxn*(yn-wn)/(xn-yn))
        xn = wn - fwn/kwn
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# a Three-Step Iterative Method to Solve a Nonlinear Equation via an Undetermined Coefficient Method
def FitriyaniImranSyamsudhuha(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k:int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn = df(xn)
        yn = xo - (2.0/3.0) * fxn/dfxn
        dfyn = df(yn)
        dfxodfy = dfyn/dfxn
        zn1 = 1.0 + (21.0/8.0)*(dfxodfy)
        zn2 = (-9.0/2.0)*(dfxodfy)*(dfxodfy)
        zn3 = (15.0/8.0)* np.power((dfxodfy),3)
        zn = xn - (zn1 + zn2 + zn3) * (fxn/dfxn)
        fzn = f(zn)
        alpha = zn-xn
        beta = yn-xn
        mu = alpha * (beta*beta - 3.0*beta*alpha + alpha*alpha)
        xn1 = alpha*beta*(-beta+alpha)*fzn
        xn2 = mu*dfxn + np.power(-alpha,3)*dfyn + 2.0*beta*(2.0*alpha-beta)*(fzn-fxn)
        xn = zn - xn1/xn2
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# a New approach for Solving Nonlinear Equations by Using of Integer Nonlinear Programming
def GhaneKanafiKordrostami(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k:int = 10,
    tol: np.float128 = 1e-15
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
        dfyn = df(yn)
        zn = xn - (2.0*fxn)/(dfxn+dfyn)
        fzn = f(zn)
        wn = zn - (fzn/dfxn) * ((dfxn + dfyn)/(3.0*dfyn-dfxn))
        fwn = f(wn)
        dfwn1 = (fxn*(dfxn+dfyn) * (-3.0*dfyn+dfxn))
        dfwn2 = ((fxn-fzn)* np.power(dfxn,3) + (-6.0*dfyn*fxn-fzn*dfyn)*(dfxn*dfxn))
        dfwn3 = ((fzn*dfyn*dfyn + 9.0*fxn*dfyn*dfyn) * dfxn + (fzn*dfyn**3))
        dfwn = -1.0 / (dfwn1*(dfwn2+dfwn3))
        xn = wn - fwn / dfwn
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Improved Higher Order Compositions for Nonlinear Equations
def Deepargyros(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k:int = 10,
    tol: np.float128 = 1e-15
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
        dfyn = df(yn)
        zn = xn - (1/2) * (1/dfxn + 1/dfyn) * fxn
        fzn = f(zn)
        xn = zn - (2*fzn*dfyn) / (2*dfxn*dfyn + dfyn**2 - dfxn**2)
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Four-Point Optimal Sixteenth-Order Iterative Method for Solving Nonlinear Equations
def UllahalFhaidahmad(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k:int = 10,
    tol: np.float128 = 1e-15
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
        t1 = fyn/fxn
        zn = yn - (1.0+2.0*t1-t1**2)/(1.0-6.0*t1**2) * (fyn/dfxn)
        fzn = f(zn)
        t2 = fzn/fxn
        t3 = fzn/fyn
        wn = zn - (1.0-t1+t3) / (1.0-3.0*t1+2.0*t3-t2) * (fzn/dfxn)
        fwn = f(wn)
        t4 = fwn/fxn
        t5 = fwn/fzn
        t6 = fwn/fyn
        q1 = 1.0 / (1.0-2.0*(t1 + t1**2 + t1**3 + t1**4 + t1**5 + t1**6 + t1**7))
        q2 = (4.0*t3) / (1.0-(31.0/(4.0*t3)))
        q3 = t2 / (1-t2-20.0*t2**3)
        q4 = np.float128((8.0*t4) / (1.0-t4) + (2.0*t5) / (1.0-t5) + t6 / (1.0-t6))
        q5 = (15.0*t1*t3) / (1.0-(131.0/(15.0*t3)))
        q6 = (54.0*t1**2*t3) / (1.0-t1**2*t3)
        q7_1 = 7.0*t2*t3 + 2.0*t1*t6 + 6.0*t6*t1**2 + 188.0*t3*t1**3
        q7_2 = 18.0*t6*t1**3 + 9.0*t2**2*t3 + 648.0*t1**4*t3
        q7 = q7_1 + q7_2
        xn = wn - (q1+q2+q3+q4+q5+q6+q7) * (fwn/dfxn)
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Modified abbasbandy’s method free from second derivative for solving nonlinear equations
def SabaNaseemSaleem(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k:int = 10,
    tol: np.float128 = 1e-15
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
        dfyn = df(yn)
        wn = yn - fyn / dfyn
        fwn = f(wn)
        dfwn = df(wn)
        xn1 = (dfyn*fwn*fwn)/(2.0* np.power(dfwn,3))
        xn2 = (dfyn-dfwn)/(fyn) * 1.0-(dfyn*fwn)/(3.0*fyn*dfwn)
        xn3 = (dfxn*fwn*(dfxn-dfyn)) / (3.0*fxn*fyn*dfwn)
        xn = wn - fwn/dfwn - xn1 * (xn2+xn3)
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# a New Steffesen-Homeier iterative method for solving nonlinear equations
def PiscoranMiclaus(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k:int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break;
        dfxn = df(xn)
        wn = xn - fxn / dfxn
        xn = xo - (fxn + f(wn))/dfxn \
            + (f(wn)*fxn) / (df(wn) + dfxn)
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# a new sixth-order scheme for nonlinear equations
def ChunNeta(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k:int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn = df(xn)
        wn = xo - fxn/dfxn
        fwn = f(wn)
        zn = wn - np.float128(fwn/dfxn \
            * (1.0/(1.0-fwn/fxn)**2))
        fzn = f(zn)
        xn = np.float128(zn - fzn/dfxn \
            * (1.0/(1.0-fwn/fxn-fzn/fxn)**2))
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# a multi-point iterative method for solving nonlinear equations with optimal order of convergence
def SalimiNikLongSharifiPansera(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k:int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    beta = 1.0
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn = df(xn)
        yn = xo - fxn/dfxn
        fyn = f(yn)
        zn1 = np.float128(yn - (2.0*fyn) / dfxn)
        zn2 = np.float128(fyn * (fxn+(beta-2.0)*fyn) \
            / (dfxn*(fxn+beta*fyn)))
        zn3 = np.float128(((dfxn*fyn) \
            / (fxn*(fxn+beta*fyn))) * (fyn/dfxn)**2)
        zn = zn1 + zn2 - zn3
        fzn = f(zn)
        tn = fyn/fxn
        un = fzn/fxn
        eta = (1.0+tn-4.0*beta*tn**3) / (1.0+tn+8.0*tn**3)
        varphi = 3.0 - 2.0/(1.0+un)
        fzy = np.float128((fyn-fzn) / (yn-zn))
        fzx = np.float128((fxn-fzn) / (xn-zn))
        fzxx = (fzx-dfxn)/(zn-xn)
        xn = zn - (fzn*eta*varphi) / (fzy + (zn-yn)*fzxx)
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# a new class of optimal four-point methods with convergence order 16 for solving nonlinear equations
def SharifiSalimiSiegmundLotfi_equ_M7(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k:int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn = df(xn)
        yn = np.float128(xo - fxn/dfxn)
        fyn = f(yn)
        tn = np.float128(fyn / fxn)
        zn = np.float128(yn - ((1.0+tn**2) * (1.0+2.0*tn+2.0*tn**2) \
            + tn**2*(2.0-8.0*tn-2.0*tn**2)) * fyn/dfxn)
        fzn = f(zn)
        un = np.float128(fzn / fxn)
        sn = np.float128(fzn / fyn)
        wn = np.float128(zn - (4.0*un-5.0*sn+(6.0+sn**3) \
            * (tn**2+sn) + (1.0+un**3)*(1.0+2.0*tn)) * fzn/dfxn)
        fwn = f(wn)
        pn = np.float128(fwn / fxn)
        qn = np.float128(fwn / fyn)
        rn = np.float128(fwn / fzn)
        xn1 = np.float128((1.0+tn) * (2.0*tn+tn**3) + 4.0*tn**2 \
            - tn**3 - tn**4 - 2.0*sn**2)
        xn2 = np.float128(2.0*tn*rn + 2.0*sn*un + 24.0*tn**4 + tn*un)
        xn3 = np.float128((2.0*tn**3-10.0*tn*un**2+6.0*tn**2*un) \
            / (1.0+2.0*tn*un) + ((1.0+2.0*pn+2.0*qn)/(1.0-rn)) \
            + (6.0*pn)/(1.0+qn))
        xn4 = np.float128((2.0*un+6*un**2)/(1.0+un) + (sn+2.0*sn**2) \
            / (1.0+sn**2) + (6.0*tn**2*rn+6.0*tn**3*rn-4.0*sn**2*un) \
            / (1.0+tn))
        xn = np.float128(wn - (xn1-xn2+xn3-xn4) * fwn/dfxn)
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# algorithm for forming derivative-free optimal methods
def KhattriSteihaug_equ_M8(
    f: Callable,
    x0: np.float128,
    k:int = 20,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    alpha = 0.01
    np.seterr(divide = 'ignore')
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol):
            break
        yn = np.float128(xo - alpha * (fxn**2/((f(xn+alpha*fxn))-fxn)))
        fyn = f(yn)
        zn1_1 = xn-yn+alpha*fxn
        zn1_2 = (xn-yn)*alpha
        zn1 = np.float128(zn1_1 / zn1_2)
        zn2_1 = xn-yn*f(xn+alpha*fxn)
        zn2_2 = (xn-yn+alpha*fxn)*alpha*fxn
        zn2 = np.float128(zn2_1 / zn2_2)
        zn3_1 = (2.0*xn-2.0*yn+alpha*fxn)*fyn
        zn3_2 = (xn-yn)*(xn-yn+alpha*fxn)
        zn3 = np.float128(zn3_1 / zn3_2)
        zn = np.float128(yn - fyn / (zn1 - zn2 - zn3))
        fzn = f(zn)
        h1_1 = np.float128(-(yn-zn)*(xn+alpha*fxn-zn))
        h1_2 = np.float128((xn-zn)*alpha*(xn-yn))
        h1 = np.divide(h1_1, h1_2)
        h2_1 = np.float128((yn-zn)*(xn-zn)*f(xn+alpha*fxn))
        h2_2 = np.float128((xn+alpha*fxn-zn)*(xn+alpha*fxn-yn)*alpha*fxn)
        h2 = np.divide(h2_1, h2_2)
        h3_1 = np.float128((xn-zn)*(xn+alpha*fxn-zn)*fyn)
        h3_2 = np.float128((yn-zn)*(xn-yn+alpha*fxn)*(xn-yn))
        h3 = np.divide(h3_1, h3_2)
        h4_1 = np.float128((xn*alpha-2.0*alpha*zn+alpha*yn)*fxn+xn**2+(-4.0*zn+2.0*yn)*xn+3.0*zn**2-2.0*yn*zn)
        h4_2 = np.float128((yn-zn)*(xn-zn)*(xn-zn+alpha*fxn))
        h4 = np.divide(h4_1, h4_2) * fzn
        hn = np.float128(h1+h2+h3+h4)
        xn = zn - fzn / hn
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Three-step iterative methods with eighth-order convergence for solving nonlinear equations
def BiRenWu_equ_3(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k:int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    alpha = 1.0
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn = df(xn)
        yn = xo - fxn/dfxn
        fyn = f(yn)
        zn = yn  - ((2.0*fxn - fyn) / (2.0*fxn - 5.0*fyn)) \
            * (fyn/dfxn)
        fzn = f(zn)
        fzx = np.float128((fzn-fxn)/(zn-xn))
        fzy = np.float128((fzn-fyn)/(zn-yn))
        fzxx = np.float128((fzx-dfxn)/(zn-xn))
        xn1 = pow((fxn/(fxn-alpha*fzn)),2.0/alpha)
        xn2 = fzn / (fzy + fzxx*(zn-yn))
        xn = zn - xn1*xn2
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Seventh and Twelfth-Order Iterative Methods for Roots of Nonlinear Equations
def Bawazir(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
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
        dfyn = df(yn)
        zn = yn - fyn/dfyn * (1.0 + (fyn*(dfxn-dfyn))/(2.0*fxn*dfyn))
        fzn = f(zn)
        dfzn = df(zn)
        xn = zn - fzn/dfzn * (1.0 + (fzn*(dfyn-dfzn))/(2.0*fyn*dfzn))
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# a variant of Newton’s method based on Simpson’s three-eighths rule for nonlinear equations
# peu précise par rapport à celle dessous
def ChenKincaidLin(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xo)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn = df(xn)
        hxn = fxn/dfxn
        xn1 = dfxn + 3*df(xn-(1/3)*hxn)
        xn2 = 3*dfxn + 3*df(xn-(2/3)*hxn) + df(xn-hxn)
        xn = xn - (8*fxn) / (xn1+xn2)
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Quadrature Rule Based Iterative Method for the Solution of Non-Linear Equations
def QureshiBozdarPirzadaarain(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k:int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break;
        dfxn = df(xn)
        xn1 = dfxn+2.0*df(xn+fxn)
        xn2 = df(xn+2.0*fxn)
        xn = xo - (4*fxn) / (xn1+xn2)
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# New Family of Iterative Methods for Solving Nonlinear Models
def aliaslamalianwarNadeem(
    f: Callable,
    df: Callable,
    d2f: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn = df(xn)
        d2fxn = d2f(xn)
        yn = xo - fxn/dfxn
        fyn = f(yn)
        #fzn = f(zn)
        zn = yn - (8.0*fyn) / (2.0*dfxn+6.0*df((xn+2*yn)/3.0) + (yn-xn)*d2fxn+3.0*(yn-xn)*d2f((xn+2*yn)/3.0))
        xn = zn - (8.0*fyn) / (2.0*dfxn+6.0*df((xn+2*zn)/3.0) + (zn-xn)*d2fxn+3.0*(zn-xn)*d2f((xn+2*zn)/3.0))
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Sixteenth-order method for nonlinear equations
def LiMuMaWang(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k:int = 10,
    tol: np.float128 = 1e-15
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
        zn1 = 2.0 * fxn - fyn
        zn2 = 2.0 * fxn - 5.0 * fyn
        zn = yn - (zn1/zn2) * (fyn/dfxn)
        fzn = f(zn)
        dfzn = df(zn)
        xn1 = 2.0*fzn - f(zn - fzn/dfzn)
        xn2 = 2.0*fzn - 5.0*f(zn - fzn/dfzn)
        xn = zn - fzn/dfzn - ((xn1)/(xn2)) * (f(zn - fzn/dfzn)/dfzn)
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# New family of eighth-order methods for nonlinear equation
def ZhangZhangDing(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn = df(xn)
        yn = xn - fxn/dfxn
        fyn = f(yn)
        u = fyn/(fxn-2.0*fyn)
        zn = yn - u*(fxn/dfxn)
        fzn = f(zn)
        u1 = fyn/(fyn-2.0*fzn)
        # u2 = fzn/(fyn-beta*fzn)
        u2 = fzn/(fyn-fzn) # beta = 0
        lmbda = u*u + u1*(2.0-u1+2.0*u) + u2
        xn = zn - lmbda*(fzn/dfxn)
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Some Higher-Order Families of Methods for Finding Simple Roots of Nonlinear Equations
def BiazarGhanbari_equ_13(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
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
        dfyn = df(yn)
        xn1 = fxn / dfxn
        xn2 = (1.0/4.0) - (7.0/2.0)*(dfyn/dfxn) + (5.0/4.0)*((dfyn*dfyn)/(dfxn*dfxn))
        xn3 = fyn / dfxn
        xn = xn - xn1 - xn2*xn3
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# an Optimal Family of Eighth-Order Iterative Methods with an Inverse Interpolatory Rational Function Error Corrector for Nonlinear Equations
def KimBehlMotsaEq33(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
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
        zn = yn - ((2.0*fxn - fyn) * fyn) / (dfxn * (2.0*fxn-5.0*fyn))
        fzn = f(zn)
        fzx = (fxn - fzn) / (xn - zn)
        xn11 = fzn * (fzn**2/fxn**2 - (3.0*fyn**3) / (2.0*fxn**3))
        xn12 = fzn * ((31.0*fyn**4) / (4.0*fxn**4) - (fyn**2+fzn**2) / dfxn**2)
        xn1 = xn11 - xn12
        xn2 = fzn * (((fyn**2+fzn**2) / dfxn**2) + (fyn**2+fyn*fzn+fzn**2)/fyn**2)
        xn3 = -dfxn + 2.0*fzx
        xn - zn - xn1 / xn3 - xn2 / xn3
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# an Optimal Family of Eighth-Order Iterative Methods with an Inverse Interpolatory Rational Function Error Corrector for Nonlinear Equations
def KimBehlMotsaEq34(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn = np.float128(df(xn))
        yn = np.float128(xo - fxn / dfxn)
        fyn = np.float128(f(yn))
        zn = np.float128(xn - (fxn**2 - fyn**2) / (dfxn*fxn - dfxn*fyn))
        fzn = np.float128(f(zn))
        xn11 = np.float128((4.0*fzn)/fxn - (2.0*fyn**2) / fxn**2)
        xn12 = np.float128((6.0*fyn**3) / fxn**3)
        xn13 = np.float128((fxn**2 + fyn**2)**2 / (fxn**2 * (fxn-fyn))**2)
        xn = np.float128(zn - (fzn*(xn11-xn12+xn13+fzn/fyn)) / dfxn)
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# a Six-order Variant of Newton’s Method for Solving Nonlinear Equations
def KumarSingh(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
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
        dfyn = df(yn)
        zn = xn - (fxn*(dfxn+dfyn)) / (dfxn**2+dfyn**2)
        fzn = f(zn)
        xn =  zn - (fzn)*(dfxn**2+dfyn**2) / (2.0*dfxn*dfyn)
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

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
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Computing Simple Roots by an Optimal Sixteenth-Order Class
# def SoleymaniShateyiSalmaniEqu15(f, df, x0, k, err) -> np.float128:
#     xo = x0
#     xn = xo
#     t = 2.0
#     while (k != 0):
#         fxn = f(xn)
#         dfxn = df(xn)
#         yn = xo - fxn/dfxn
#         fyn = f(yn)
#         zn = yn - (fxn+t*fyn)/(fxn+(t-2.0)*fyn) * (fyn/dfxn)
#         fzn = f(zn)
#         fyx = np.float128((fxn-fyn)/(xn-yn))
#         fzx = (fxn-fzn)/(xn-zn)
#         d1 = 1.0/((fyn-fxn)*(fyn-fzn)*fyx)
#         d2 = 1.0/((fzn-fxn)*(fyn-fzn)*fzx)
#         d3 = 1.0/(dfxn*(fzn-fxn)*(fyn-fzn))
#         d4 = 1.0/(dfxn*(fyn-fxn)*(fyn-fzn))
#         d = d1-d2+d3-d4
#         c = 1.0/((fyn-fxn)*fyx) - 1.0/(dfxn*(fyn-fxn)*fyx) - d*(fyn-fxn)
#         wn = yn * c*fxn**2 - d*fxn**3
#         fwn = f(wn)
#         fwx = (fxn-fwn)/(xn-wn)
#         varphiw = 1.0/(fwx*(fwn-fxn)) - 1.0/(dfxn*(fwn-fxn))
#         varphiy = 1.0/(fyx*(fyn-fxn)) - 1.0/(dfxn*(fyn-fxn))
#         varphiz = 1.0/(fzx*(fzn-fxn)) - 1.0/(dfxn*(fzn-fxn))
#         gp = ((varphiw-varphiz) / (fwn-fzn) - (varphiy-varphiz) / (fyn-fzn)) / (fwn-fyn)
#         dp = (varphiw-varphiz) / (fwn-fzn) - gp*(fwn-2.0*fxn+fzn)
#         cp = varphiw - dp*(fwn-fxn) - gp*(fwn-fxn)**2
#         xn = yn + cp*fxn**2 - dp*fxn**3 + gp*fxn**4
#         if (fabs(xn-xo) <= err or fabs(fxn) <= err):
#             break
#         else:
#             xo = xn
#             k -= 1
#     return xn

# Optimal derivative-free root finding methods based on the Hermite interpolation
def akramZafarYasmin_MFM16(
    f: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        wn = xn + fxn**4
        fwn = f(wn)
        fwx = np.float128((fxn-fwn) / (xn-wn))
        yn = xo - fxn / fwx
        fyn = f(yn)
        fyx = (fyn-fxn) / (yn-xn)
        zn = yn - fyn / (2.0*fyx - fwx)
        fzn = f(zn)
        fzx = (fzn-fxn) / (zn-xn)
        k3_1_1 = np.float128(fzx * (2.0+(zn-xn) / (zn-yn)))
        k3_1_2 = np.float128(((zn-xn)**2 / ((yn-xn)*(zn-yn))) * fyx)
        k3_1_3 = np.float128(fwx * ((zn-yn) / (yn-xn)))
        k3 = np.float128(k3_1_1 - k3_1_2 + k3_1_3)
        tn = zn - fzn / k3
        ftn = f(tn)
        ftz = (ftn-fzn) / (tn-zn)
        fzy = (fyn-fzn) / (yn-zn)
        ftzy = (fzy-ftz) / (tn-yn)
        fzyx = (fyx-fzy) / (xn-zn)
        ftzyx = (ftzy - fzyx) / (xn-tn)
        ftzyx2_1 = 1.0/((tn-xn)**2*(tn-yn))*(ftz-fzy) - 1.0/((tn-xn)**2*(zn-xn))*(fzy-fyx)
        ftzyx2_2 = 1.0/((tn-xn)*(zn-xn)**2)*(fzy-fyx) + 1.0/((tn-xn)*(zn-xn)*(yn-xn))*(fyx-fwx)
        ftzyx2 = ftzyx2_1 - ftzyx2_2
        k4 = ftz + (tn-zn) * ftzy + ((tn-zn) * (tn-yn)) * ftzyx + ((tn-zn)*(tn-yn)*(tn-xn))*ftzyx2
        xn = tn - ftn / k4
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Optimal Sixteenth Order Convergent Method Based on Quasi-Hermite Interpolation for Computing Roots
def ZafarHussainFatimahKharalM2(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
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
        zn = yn - (fyn/dfxn) * (fxn / (fxn-2*fyn))
        fzn = f(zn)
        fyx = np.float128((fxn-fyn)/(xn-yn))
        fyxx = 2.0 * (fyx-dfxn)/(xn-yn)
        fxz = (fzn-fxn) / (zn-xn)
        fyz = (fzn-fyn) / (zn-yn)
        fxy = (fyn-fxn) / (yn-xn)
        tn = zn - fzn * (1/ (2*fxz+fyz-2*fxy + (yn-zn)*fyxx))
        ftn = f(tn)
        ftz = (fzn-ftn) / (zn-tn)
        fty = (fyn-ftn) / (yn-tn)
        fzy = (fyn-fzn) / (yn-zn)
        ftzy = (ftz-fty) / (tn-yn)
        ftzyx = ((ftz-fzy) / ((tn-xn)*(tn-yn))) - ((fzy-fyx) / ((tn-xn)*(zn-xn)))
        ftzyx2_1 = (ftz-fzy)/((tn-xn)**2*(tn-yn)) - (fzy-fyx)/((tn-xn)**2*(zn-xn))
        ftzyx2_2 = (fzy-fyx)/((zn-xn)**2*(tn-xn)) + (fyx-dfxn)/((tn-xn)*(zn-xn)*(yn-xn))
        ftzyx2 = ftzyx2_1 - ftzyx2_2
        h4 = ftz + (tn-zn) * ftzy + (tn-zn) * (tn-yn) * ftzyx + (tn-zn)*(tn-yn)*(tn-xn)*ftzyx2
        xn = tn - ftn / h4
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# a biparametric family of optimally convergent sixteenth-order multipoint methods with their fourth-step weighting function as a sum of a rational and a generic two-variable function
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
    beta = 2.0
    sigma = -2.0
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn = df(xn)
        yn = xo - fxn/dfxn
        fyn = f(yn)
        un = fyn/fxn
        kf = (1.0+beta*un+(-9.0+((5.0*beta)/2.0))*un**2) \
            / (1.0 + (beta-2.0)*un + (-4.0+beta/2.0)*un**2)
        zn = yn - kf*(fyn/dfxn)
        fzn = f(zn)
        vn = fzn/fyn
        wn = fzn/fxn
        hf = (1.0+2.0*un+(2.0+sigma)*wn) / (1.0 - vn+sigma*wn)
        sn = zn - hf*(fzn/dfxn)
        fsn = f(sn)
        tn = fsn/fzn
        phi1 = 11.0*beta**2-66.0*beta+136.0
        phi2 = 2.0*un * (sigma**2-2*sigma-9.0)-4.0*sigma-6.0
        guw = (-1.0/2.0) * (un*wn * (6.0+12.0*un+un**2+un**2 \
            * (24.0-11.0*beta) + un**3*phi1+4.0*sigma))+phi2*wn**2
        wf = (1.0 + 2.0*un+(2.0+sigma)*vn*wn) / (1.0-vn-2.0*wn \
            - tn+2.0*(1.0+sigma)*vn*wn) + guw
        xn = sn - wf*(fsn/dfxn)

        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# an optimal sixteenth order convergent method to solve nonlinear equations
def EsmaeiliahmadiErfanifar(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k:int = 10,
    tol: np.float128 = 1e-15
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
        zn = np.float128(yn - (fxn/(fxn-2.0*fyn)) * (fyn/dfxn))
        fzn = f(zn)
        fxy = (fxn-fyn)/(xn-yn)
        fxz = (fxn-fzn)/(xn-zn)
        fyz = (fyn-fzn)/(yn-zn)
        wn = np.float128(zn - (fxn+fzn)/fxn * ((fxy*fzn) / (fxz*fyz)))
        fwn = f(wn)
        fxw = (fxn-fwn)/(xn-wn)
        fzw = (fzn-fwn)/(zn-wn)
        an = np.float128(fwn/(fzn*fyn))
        bn = fyn**3/fxn**4
        cn = fzn/fxn**2 - fyn**3/fxn**4
        un = fwn/(fxn*fzn)
        vn = (fyn*fzn)/fxn**3
        sn = (fzn-fyn**3/fxn**2)*(fyn/fxn**3)
        tn = (fzn/fyn-fyn**2/fxn**2)**2 * (1.0/fxn)
        fzx = (fzn-fxn)/(zn-xn)
        fxw = (fxn-fwn)/(xn-wn)
        fzw = (fzn-fwn)/(zn-wn)
        fzxx = (fzx-dfxn)/(zn-xn)
        g = an-3.0*bn-4.0*cn
        h = un-6.0*vn-6.0*sn-2.0*tn
        xn1 = fwn / (2.0*fxw+fzw-2.0*fxz+(zn-wn)*fzxx)
        xn2 = (fwn*fzn)/dfxn * (g+2.0*h)
        xn = wn - xn1-xn2
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Six order iterative methods for solving nonlinear equations
def Saeed(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
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
        dfyn = df(yn)
        d = (2.0*df((3.0*xn+yn)/4.0) - df((xn+yn)/2.0) + 2.0*df((xn+3.0*yn)/4.0))
        zn = yn - (3.0*fxn) / d
        fzn = f(zn)
        xn = zn - (fzn * d) / (dfxn * (3.0*dfyn - 3.0*dfxn + d))
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Some Real-Life applications of a Newly Designed algorithm for Nonlinear Equations and Its Dynamics via Computer Tools
def NaseemRehmanYounis(
    f: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        gn = np.float128(f(xn+fxn)/fxn)
        fgn = f(gn)
        vn = xo - fxn/fgn
        fvn = f(vn)
        hn = np.float128((fvn-fxn) / (vn-xn))
        fhn = f(hn)
        wn = vn - fvn/fhn
        fwn = f(wn)
        xn = wn - (fwn*fvn) / (fhn*(fvn-2.0*fwn))
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# a New Family of Fourth-Order Optimal Iterative Schemes and Remark on Kung and Traub’s Conjecture
def Com69(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
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
        f1 = (fxn/(fxn-fyn))**2.0 * (fyn/dfxn)
        f2 = (2.0*fxn*(fxn-fyn))/(dfxn*(fxn-2.0*fyn))
        xn = xn - fxn/dfxn + f1 - f2
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Solution of nonlinear equations using three point Gaussian quadrature formula and decomposition technique
def SanaaslamNoorInayatNoor(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    s35 = np.sqrt(3.0/5.0)
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn = df(xn)
        yn = xo - fxn/dfxn
        fyn = f(yn)
        ypxdd = (yn+xn)/2.0
        ymxdd = (yn-xn)/2.0
        zn1 = 5.0*df(ypxdd - ymxdd * s35)
        zn2 = 8.0*df(ypxdd)
        zn3 = 5.0*df(ypxdd + ymxdd * s35)
        zn = (18.0*fyn) / (zn1+zn2+zn3)
        fzn = f(zn)
        zpxdd = (zn+xn)/2.0
        zmxdd = (zn-xn)/2.0
        wn1 = 5.0*df(zpxdd - zmxdd * s35)
        wn2 = 8.0*df(zpxdd)
        wn3 = 5.0*df(zpxdd + zmxdd * s35)
        wn = (18.0*fzn) / (wn1+wn2+wn3)
        fwn = f(wn)
        wpxdd = (wn+xn)/2.0
        wmxdd = (wn-xn)/2.0
        xn1 = 5.0*df(wpxdd - wmxdd * s35)
        xn2 = 8.0*df(wpxdd)
        xn3 = 5.0*df(wpxdd + wmxdd * s35)
        xn = wn - (18.0*fwn) / (xn1+xn2+xn3)
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# New Predictor-Corrector Iterative Methods with Twelfth-Order Convergence for Solving Nonlinear Equations
def abdulHassan(
    f: Callable,
    df: Callable,
    d2f: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn = df(xn)
        yn = xo - (2.0*fxn)/(3.0*dfxn)
        dfyn = df(yn)
        zn = xn - ((3.0*dfyn + dfxn) / (6.0*dfyn - 2.0*dfxn)) * (fxn/dfxn)
        fzn = f(zn)
        dfzn = df(zn)
        d2fzn = d2f(zn)
        xn1 = (fzn*dfzn*(2.0+2.0*dfzn**2 + fzn*d2fzn))
        xn2 = (2.0*dfzn**2 * (1.0+dfzn**2) - fzn*d2fzn)
        xn = fzn - xn1/xn2
        print("xn=", xn)
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# TWELFTH-ORDER METHOD FOR NONLINEaR EQUaTIONS
def HouLi(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k:int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    alpha = 1.0
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn = df(xn)
        yn = xo - fxn/dfxn
        fyn = f(yn)
        zn = yn - ((2.0*fxn-fyn)/(2.0*fxn-5.0*fyn)) * (fyn/dfxn)
        fzn = f(zn)
        fzy = np.float128((fyn-fzn) / (yn-zn))
        fzx = np.float128((fzn-fxn) / (zn-xn))
        fzxx = np.float128((fzx-dfxn) / (zn-xn))
        fx = fzy + fzxx*(zn-yn)
        wn = zn - ((2.0*fxn-fzn)/(2.0*fxn-5.0*fzn)) * (fzn/fx)
        fwn = f(wn)
        xn = wn - ((fxn+(2.0+alpha)*fzn) / (fxn+alpha*fzn)) * (fwn/fx)
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# a New Family of Nonlinear Fifth-Order Solvers for Finding Simple Roots
def Ghanbari(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn = df(xn)
        op = fxn/dfxn
        yn = xo - op
        fyn = f(yn)
        dfyn = df(yn)
        xn1 = ((3.0*dfxn+dfyn) / (-dfxn+5.0*dfyn))
        xn2 = ((2.0*fyn+dfxn) / (fyn+dfxn))
        xn3 = (fyn / fyn+dfxn)
        xn = xn - op - xn1 * xn2 * xn3
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Tenth-Order Iterative Methods without Derivatives for Solving Nonlinear Equations
def alHusaynialSubaihiYSM4(
    f: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        gn = np.float128(xn+fxn**3)
        fgn = np.float128(f(gn))
        fgx = np.float128((fgn-fxn) / (gn-xn))
        yn = np.float128(xn - fxn/fgx)
        fyn = np.float128(f(yn))
        fgy = np.float128((fgn-fyn) / (gn-yn))
        zn = np.float128(yn - ((fxn-fgn)/(fyn-fxn)) * (1.0/fgx - 1.0/fgy))
        fzn = np.float128(f(zn))
        fyx = np.float128((fyn-fxn) / (yn-xn))
        tn = np.float128(f(zn - fzn/(2.0*fyx-fgx))/fzn)
        xn1 = np.float128(fzn/(2.0*fyx-fgx))
        xn2_1 = np.float128(zn - fzn/(2.0*fyx-fgx))
        xn2_2 = np.float128(2.0*fyx-fgx)
        xn2 = np.float128((1.0-tn)*(f(xn2_1)/xn2_2))
        xn = np.float128(zn - xn1 - xn2)
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Excellent Higher Order Iterative Scheme for Solving Non-linear Equations
def HeneritaPanday(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn = df(xn)
        yn = xo - xn/dfxn
        fyn = f(yn)
        zn = xn - (fxn*(fyn-fxn))/(dfxn*(2.0*fyn-fxn))
        fzn = f(zn)
        dfzn = df(zn)
        xn1 = ((fzn/dfzn) + (fzn*(fzn-fzn))/(fzn*(2.0*fzn-fxn)))**2
        xn2 = (1.0/2.0*dfzn) * (2.0/(xn-zn)) * (3.0 * ((fxn-fzn)/(xn-zn)) - 2.0*dfzn-dfxn)
        xn = zn - xn1*xn2
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Some novel and optimal families of King's method with eight and sixteenth-order of convergence
def MarojuBehlLMotsa(
    f: Callable,
    df: Callable,
    x0: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    beta = -1.0/2.0
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn = df(xn)
        yn = x0 - fxn/dfxn
        fyn = f(yn)
        zn = yn - ((fxn + beta*fyn) / (fxn + (beta-2.0)*fyn)) * (fyn/dfxn)
        fzn = f(zn)
        u = fzn/fyn
        v = fyn/fxn
        a1 = (2.0*(beta**2 - 6.0*beta+6.0)) / (2.0*beta-5.0)
        a2 = (2.0*beta-5.0) / a1
        guv = 1.0 + u + 4.0*u*v - ((4.0*beta+1.0)*v) / (2.0*(beta**2 - 6.0*beta+6.0)) + (a2*v)/(a1*v+1.0)
        tn = zn - (fzn / dfxn) * guv
        ftn = f(tn)
        an = xn-zn
        bn = tn-xn
        cn = tn-zn
        u1 = ftn*(bn**2*dfxn + bn*fxn - cn*fzn) + an*(fxn-an*dfxn)*fzn
        u2 = an*bn*cn*dfxn * (fyn-fxn) + cn*fyn*fxn*(an-bn)
        v1_1 = fyn*(bn*ftn*(bn**2*dfxn + bn*fxn - cn*fzn))
        v1_2 = fyn*((an**3*dfxn + cn*an*ftn - an**2*fxn)*fzn)
        v1 = v1_1+v1_2
        v2_1 = an**2*bn**2*cn*dfxn**2 * (2.0*fyn-fxn)
        v2_2 = an*bn*cn*(2.0*an-cn)*dfxn*fyn*fxn
        v2_3 = cn*(an*bn-an*cn-bn**2)*fyn*fxn**2
        v2 = v2_1 + v2_2 + v2_3
        theta5_n = an*bn*(u1*fxn**2*fyn + u2*dfxn*ftn*fzn)
        theta5_d = v1*fxn**3 + v2*dfxn*ftn*fzn
        theta5 = theta5_n / theta5_d
        xn = xn - theta5 * fxn
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# On a general class of optimal order multipoint methods for solving nonlinear equations
def SharmaargyrosKumar(
    f: Callable,
    df: Callable,
    x0: np.float64,
    k:int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn = df(xn)
        det1 = 1.0
        delta1 = df(xn)
        wn = xo - det1/delta1 * fxn
        fwn = f(wn)
        fxw = (fwn-fxn)/(wn-xn)
        det2 = 1.0*fxn - 1.0*fwn
        delta2 = dfxn*fwn - fxw*fxn
        zn = xn - det2/delta2 * fxn
        fzn = f(zn)
        fxz = (fzn-fxn)/(zn-xn)
        m3 = np.array([[1.0,fxn,xn*fxn],
                        [1.0,fwn,wn*fwn],
                        [1.0,fzn,zn*fzn]], dtype=np.float64)
        det3 = np.linalg.det(m3)
        md3 = np.array([[dfxn,fxn,xn*fxn],
                            [fxw,fwn,wn*fwn],
                            [fxz,fzn,zn*fzn]], dtype=np.float64)
        delta3 = np.linalg.det(md3)
        un = xn - det3/delta3 * fxn
        fun = f(un)
        fxu = (fun-fxn)/(un-xn)
        m4 = np.array([[1.0,fxn,xn*fxn,xn**2**fxn],
                        [1.0,fwn,wn*fwn,wn**2*fwn],
                        [1.0,fzn,zn*fzn,zn**2*fzn],
                        [1.0,fun,un*fun,un**2*fun]], dtype=np.float64)
        det4 = np.linalg.det(m4)
        md4 = np.array([[dfxn,fxn,xn*fxn,xn**2**fxn],
                        [fxw,fwn,wn*fwn,wn**2*fwn],
                        [fxz,fzn,zn*fzn,zn**2*fzn],
                        [fxu,fun,un*fun,un**2*fun]], dtype=np.float64)
        delta4 = np.linalg.det(md4)
        xn = np.float128(xn - det4/delta4 * fxn)
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

#Sixteenth-order optimal iterative scheme based on inverse interpolatory rational function for nonlinear equations
# def SalimiBehl(f, df, x0, k, err) -> np.float128:
#     xo = x0
#     xn = xo
#     while k != 0:

# New Sixteenth-Order Derivative-Free Methods for Solving Nonlinear Equations
def Thukral(
    f: Callable,
    x0: np.float128,
    k:int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        wn = xo + fxn
        fwn = f(wn)
        fwx = np.float128((fwn-fxn)/(wn-xn))
        yn = xn - (fxn/fwx)
        fyn = f(yn)
        fxy = np.float128((fxn-fyn)/(xn-yn))
        fyw = np.float128((fyn-fwn)/(yn-wn))
        fxw = np.float128((fxn-fwn)/(xn-wn))
        phi3 = fxw/fyw
        zn = yn - phi3*(fyn/fxy)
        fzn = f(zn)
        u1 = fzn/fxn
        u2 = fzn/fwn
        u3 = fyn/fxn
        u4 = fyn/fwn
        fyz = np.float128((fyn-fzn)/(yn-zn))
        fxz = np.float128((fxn-fzn)/(xn-zn))
        eta = (1.0/(1.0+2.0*u3*u4**2)) * (1.0/(1.0-u2))
        an = zn-eta*(fzn/(fyz-fxy+fxz))
        fan = f(an)
        u5 = fan/fxn
        u6 = fan/fwn
        sigma = 1.0+u1*u2 - u1*u3*u4**2 + u5 + u6 \
            + u1**2*u4 + u2**2*u3 + 3.0*u1*u4**2 \
            * (u3**2-u4**2) * (1.0/fxy)
        fya = np.float128((fyn-fan)/(yn-an))
        fza = np.float128((fzn-fan)/(zn-an))
        xn = zn - sigma * (fyz*fan)/(fya*fza)

        if np.less(np.fabs(xn - xo), tol):
          break
        else:
        	xo = xn
    
    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# def Varona(
#    f: Callable,
#    df: Callable,
#     x0: np.float128,
#     k:int = 10,
#     tol: np.float128 = 1e-15
#     ) -> np.float128:
#     dec = np.finfo(np.float128).precision
#     xo = x0
#     xn = xo
#     for _ in range(k):
#         fxn = f(xn)
#         if np.isclose(fxn, 0.0, atol=tol) == True:
#             break
#         dfxn = df(xn)
#     return np.format_float_positional(np.float128(xn), \
#         unique=False, precision=dec)

# def Varona(
#    f: Callable,
#    df: Callable,
#     x0: np.float128,
#     k:int = 10,
#     tol: np.float128 = 1e-15
#     ) -> np.float128:
#     dec = np.finfo(np.float128).precision
#     xo = x0
#     xn = xo
#     for _ in range(k):
#         fxn = f(xn)
#         if np.isclose(fxn, 0.0, atol=tol) == True:
#             break
#         dfxn = df(xn)
#         yn = xo - fxn / dfxn
#         fyn = f(yn)
#         tn = fyn / fxn
#         q_t = 1 + 2*tn
#         zn = yn - qt * (fyn / dfxn)
#         fzn = f(zn)
#         w_ts = 1 + 2*tn + tn**2 - 4*tn**3 + sn + 4*tn
#         wn = zn - wts * (fzn / dfxn)
#         fwn = f(wn)
#         h_tsu = 
#         hn = zn - htsu * (fzn / dfxn)
#         fwn = f(wn)
#         sn = fzn / fyn
#         un = fwn / fzn
#         vn = fhn / fwn
#         xn = hn - jtsuv * (fwn / dfxn)
#         if np.less(np.fabs(xn-xo), tol) == True:
#             break
#         xo = xn
#     return np.format_float_positional(np.float128(xn), \
#         unique=False, precision=dec)

# Further acceleration of Thukral Third-Order Method for Determining Multiple Zeros of Nonlinear Equations
def Thukralm(
    f: Callable,
    df: Callable,
    d2f: Callable,
    d3f: Callable,
    x0: np.float128,
    m: int,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        t1 = df(xn) / fxn
        t2 = d2f(xn) / df(xn)
        t3 = d3f(xn) / d2f(xn)
        u1 = 3.0 * m * (m+1)
        u2 = 3.0 * m**2
        u3 = (2.0*m + 1) * (m + 1)
        u4 = 3.0 * m * (m+1)
        u5 = m**2
        xn = xo - (u1*t1-u2*t2)/(u3*t1**2-u4*t1*t2+u5*t2*t3)
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Constructing higher-order methods for obtaining the multiple roots of nonlinear equations
def ZhouChenSong4(
    f: Callable,
    df: Callable,
    x0: np.float128,
    m: int,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn = df(xn)
        yn = xn - ((2.0*m)/(2.0+m)) * (fxn/dfxn)
        dfyn = df(yn)
        x1 = (m**4/8.0)*((m+2.0)/m)**m * ((dfyn/dfxn)*(fxn/dfxn))
        x2 = ((m*(m+2.0))**3/8.0)*(m/(m+2.0))**m * (fxn/dfyn)
        x3 = (1.0/4.0)*m * (m**3+3.0*m**2+2.0*m-4.0) * (fxn/dfxn)
        xn = xn - x1-x2+x3
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# Computing multiple zeros by using a parameter in Newton-Secant method
def KanwarBhatiaKansal(
    f: Callable,
    df: Callable,
    x0: np.float128,
    m: int,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn = df(xn)
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

def FerraraSharifiSalimi(
    f: Callable,
    df: Callable,
    x0: np.float128,
    m: int,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        dfxn = df(xo)
        fod = fxn/dfxn
        yn = xo - fod
        fyn=f(yn)
        xn = np.float128((((-1.0+m)**(-1.0+m) * fxn) / ((-1.0+m)**(-1.0+m) * fxn - m**(-1.0+m)*fyn)) * fod)
        if np.less(np.fabs(xn-xo), tol) == True:
            break
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.722.3454&rep=rep1&type=pdf
# a third-order modification of Newton’s method for multiple roots
# Changbum Chun, Beny Neta
# def chunneta2(f, df, ddf, x0, m, k, err) -> np.float128:
#     xo = x0
#     xn = xo
#     while k != 0:
#         fxn = f(xn)
#         dfxn = df(xn)
#         ddfxn = ddf(xn)
#         n = np.float128(2*m**2*fxn**2*ddfxn)
#         d = np.float128(m*(3-m)*fxn*dfxn*ddfxn+(m-1)**2*dfxn**3)
#         print('n', n)
#         print('d', d)
#         xn = xo - n/d
    #     if (fabs(xn-xo) <= err or fabs(fxn) <= err):
    #         break
    #     else:
    #         k -= 1
    #         xo = xn
    # return xn

def HBM2(
    f: Callable,
    x0: np.float128,
    beta: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        fx = np.float128(f(xn))
        yn = np.float128(xn - (beta*fx**2) / (f(xn+beta*fx)-fx))
        fy = np.float128(f(yn))
        p1 = np.float128(2.0*((fy-fx)/(yn-xn))) - np.float128((f(xn+beta*fx)-fx)/(beta*fx))
        p2 = np.float128((2.0/(yn-xn))*(((fy-fx)/(yn-xn)) - (f(xn-beta*fx)-fx)/(beta*fx)))
        xn = np.float128(xo - (2.0*fy*p1)/(2.0*p1**2.0-fy*p2))
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

def HBM1(
    f: Callable,
    x0: np.float128,
    beta: np.float128,
    k: int = 10,
    tol: np.float128 = 1e-15
    ) -> np.float128:
    dec = np.finfo(np.float128).precision
    xo = x0
    xn = xo
    for _ in range(k):
        fxn = f(xn)
        if np.isclose(fxn, 0.0, atol=tol) == True:
            break
        yn = xo - np.float128((beta*fxn**2) / (f(xn+beta*fxn)-fxn))
        fyn = np.float128(f(yn))
        p0 = np.float128((f(xn+beta*fxn)-fxn)/(beta*fxn))
        xn = xo - np.float128((fxn**2+fyn**2)/(p0*(fxn-fyn)))
        if np.less(np.fabs(xn-xo), tol) == True:
            break
        xo = xn

    return np.format_float_positional(np.float128(xn), \
        unique=False, precision=dec)

def cerclecomplex(n: int) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, n)
    return np.exp(theta * 1j)

def aberth_initial(coefs: np.ndarray) -> np.ndarray:
    n: int = coefs.shape[0]
    z: np.ndarray = np.zeros(n, dtype = np.complex256)
    alpha: np.float128 = np.pi / (2 * n)
    r0 = np.abs(np.polyval(coefficients, alpha))**(1/n)
    theta = 0+1j

    for k in range(0,n):
        theta = k * ((2 * np.pi) / n) * (k) + alpha
        z[k] = (-coefs[1]/n) + r0 * np.complex256(np.exp(theta) * 1j)
    return z

def horner(z:np.complex256, coefs: np.ndarray) -> np.complex256:
    t = np.complex256(0+0j)
    for c in coefs:
        t *= z + c
    return t

def WeierstrassDurandKerner(
	coefficients: np.ndarray,
	it: int = 10,
	tol: np.float64 = np.finfo(np.float64).eps
	) -> np.ndarray:

	if coefficients[0] != 1.0:
		coefficients /= coefficients[0]
		
	k: int = 1
	n = len(coefficients) - 1
	if n < 3:
		raise ValueError("Degré < 3 !")
		sys.exit(1)

	# borne de Cauchy
	# up = 1 + np.max(coefficients[0:-1]) / np.abs(coefficients[0])
	# lo = np.abs(coefficients[-1]) / (np.abs(coefficients[-1]) + np.max(coefficients[0:-1]))

	x1 = -coefficients[1] / n + np.exp((2j * np.pi * np.arange(n, dtype=complex)) / n + np.pi / (2*n))
	x0 = x1.copy()
	while k <= it:
		for i in range(n):
			num = np.polyval(coefficients, x1)
			diff = x1[i] - np.delete(x1, i)
			denom = np.multiply.reduce(diff) if np.multiply.reduce(diff) != 0 else tol
			try:
				x1[i] -= num[i] / denom
			except ZeroDivisionError:
				sys.exit(1)

		if np.all(np.linalg.norm(x1 - x0) < tol) \
			or np.isnan(x1.all().real) \
			or np.isnan(x1.all().imag):
			break
		else:
			x0 = np.copy(x1)
			k += 1

	return x1


def aberthEhrlich(
    coefficients: np.ndarray,
    valeurs_initiales: np.ndarray,
    iterations: int,
    epsilon: np.float64 
    ) -> np.ndarray:

    n: int = valeurs_initiales.shape[0]
    c = np.array(coefficients, dtype = np.complex128)
    x0 = np.array(valeurs_initiales, dtype = np.complex128)
    x = np.empty_like(x0, dtype = np.complex128)
    xi: np.complex128(0+0j)
    s: np.complex128(0+0j)
    it:int = 1

    a0 = coefficients[0]

    if n < 2 or a0 == 0.0 or coefficients[-1] == 0.0:
        raise ValueError("Polynôme incorrect !")
        sys.exit(1)

    if a0 != 1.0:
        c /= a0

    # init = aberth_initial(coefficients)
    poly = Poly(coefficients[::-1])
    polyd = poly.deriv(1)
    while it < iterations:
        for i in range(n):
            s = 1+0j
            xi = x0[i]
            for j in range(n):
                if i != j:
                    s = s + 1 / (xi - x0[j])
            evalpoly = np.polyval(c, xi)
            evalpolyd = np.polyval(polyd.coef, xi)
            x[i] = xi - 1 / (evalpolyd / evalpoly - s)
        res = np.linalg.norm(x - x0)
        if np.less(res, epsilon) or it == iterations:
            break
        else:
            np.copyto(x0, x)
            it += 1
    
    print(f"{it = },\t{res = }")
    print(f"{x = }")

    return np.sort(x, axis=0)


def Nourein(
    coefficients: np.ndarray,
    it: int = 20,
    tol: np.float64 = np.finfo(np.float64).eps
    ) -> np.ndarray:

    wi = wk = np.complex128(1+0j)
    xi = xk = np.complex128(1+0j)
    den1 = den2 = np.complex128(1+0j)
    s = np.complex128(1+0j)

    if coefficients[0] != 1.0:
        coefficients /= coefficients[0]

    n = len(coefficients) - 1
    if n < 3 or coefficients[-1] == 0.0:
        raise ValueError("Polynôme incorrect !")

    x1 = -coefficients[1] / n + np.exp((2j * np.pi * np.arange(n, dtype=np.complex128)) / n + np.pi / (2*n))
    x0 = x1.copy()

    k: int = 1
    while k <= it:
        for i in range(n):
            xi = x0[i]
            for j in range(n):
                if i != j:
                    den1 = den1 * (xi - x0[j])
            num = np.polyval(coefficients, xi)
            try:
                wi = num / den1
            except ZeroDivisionError:
                sys.exit(1)
            for k in range(n):
                xk = x0[k]
                if i != k:
                    den2 = 1+0j
                    for l in range(n):
                        if k != l:
                            den2 = den2 * (xk - x0[l])
                    num = np.polyval(coefficients, xk)
                    try:
                        wk = num / den2
                    except ZeroDivisionError:
                        sys.exit(1)
                s = s + wk / (xi - wi - xk)
            x1[i] = xi - wi / (1 + s)

        if np.all(np.linalg.norm(x1 - x0) < tol) \
			or np.isnan(x1.all().real) \
			or np.isnan(x1.all().imag):
            break
        else:
            x0 = np.copy(x1)
            k += 1

    return x1


def NM10D(
    coefficients: np.ndarray,
    valeurs_initiales: np.ndarray,
    iterations: int,
    epsilon: np.float64
    ) -> np.ndarray:

    n: int = valeurs_initiales.shape[0]
    coefs = np.array(coefficients, dtype = np.complex128)
    x = np.array(valeurs_initiales, dtype = np.complex128)
    y = np.empty_like(x, dtype = np.complex128)
    z = np.copy(x)
    u = np.empty_like(x, dtype = np.complex128)
    den: np.complex128
    fx: np.complex128
    fy: np.complex128
    wy: np.complex128
    wu: np.complex128
    xi: np.complex128
    yi: np.complex128
    ui: np.complex128
    it:int = 1

    a0 = coefficients[0]

    if n < 2 or a0 == 0.0 or coefficients[-1] == 0.0:
        raise ValueError("Polynôme incorrect !")
        sys.exit(1)

    if a0 != 1.0:
        c /= a0

    while it < iterations:
        for i in range(n):
            den = 1+0j
            xi = x[i]
            for j in range(n):
                if j != i:
                    den *= (xi - z[j])
            fx = np.polyval(coefs, xi)
            try:
                wy = fx / den
            except ZeroDivisionError:
                continue
            y[i] = xi - wy
            yi = y[i]
            den = 1+0j
            for j in range(n):
                if j != i:
                    den *= (yi - y[j])
            fy = np.polyval(coefs, yi)
            try:
                wu = fy / den
            except ZeroDivisionError:
                continue
            u[i] = yi - wu
            ui = u[i]

        res = np.linalg.norm(ui - yi)
        if np.less(res, epsilon) or it == iterations:
            break
        else:
            np.copyto(z, u)
            it += 1
    
    print(f"{it = },\t{res = }")
    print(f"{np.sort(u,axis=0) = }")

    return np.sort(u, axis=0)


def bairstow (n, r, s, p0, q0, deg, coefs):
    return 0

#public boolean is_tridiag_sym() {
#    int n = this.size;
#
#    for (int i = 1; i < n-1; i++) {
#        if (this.data[i][i-1] == this.data[i][i+1]) {
#            System.out.printf("i = %0.f" + (i) + "j = " + (i-1));
#            return false;
#        }
#    }
#    System.out.print("\n");
#
#    return true;
#}
#
#public void swaprow( m, int i, int j) {
#    double temp;
#
#    for (int k = i; i < m.size; i++) {
#        temp = m.data[i][j];
#        m.data[i][k] = m.data[j][k];
#        m.data[j][k] = temp;
#    }
#}

# function Project2DTo3D(pos2D, viewDir, upDir, fov, height) {
# // height is the window’s height
# // fov is the field-of-view
# // upDir is a direction vector that points up rightPos = CrossProduct(upDir, viewDir); distance = height / tan(fov);
# return viewDir * distance 􏰀
# upDi2r * pos2D.y + rightPos * pos2D.x;
# }

def prodscal(a: np.ndarray, b: np.ndarray) -> float:
	return np.sum(np.fromiter((ai*bi for ai,bi in zip(a,b)),a.dtype))

def norme(v: np.ndarray) -> float:
    return np.sqrt(np.sum([i**2 for i in v]))

def est_ortho1(q: np.ndarray) -> bool:
	if np.allclose(np.eye(len(q)), q@q.T, atol=1e-6):
		return True
	else:
		return False

def est_ortho2(q: np.ndarray) -> bool:
	if np.linalg.norm(q[:,0]) == 1 \
	  and np.linalg.norm(q[:,1]) == 1 \
	  and q[:,0]@q[:,1] == 0:
		return True
	else:
		return False

# if __name__ == "__main__":
# 	R = np.array([[np.cos(30), -np.sin(30)], \
# 		[np.sin(30), np.cos(30)]])
# 	print("Test numéro 1 : ", est_ortho1(R))
# 	print("Test numéro 2 : ", est_ortho2(R))

def est_tridiag_nonsym(m):
    n = len(m)
    for i in range(1,n-1):
        if m[i][i-1] != m[i][i+1]:
           return False
    return True

def issquare(m):
    if (m.shape(0) == m.shape(1)):
        return True
    else:
        raise ValueError("Matrice non carrée")
        return False

def est_symetrique(m):
    # n = len(m)
    # for i in range(n):
    #     for j in range(n):
    #         if (m[i,j] == m[j,i]):
    #             return True
    #         else:
    #             return False
    if m.shape(0) == m.shape(1) and (m.T == m).all():
        return True
    else:
        return False
    
def isspd(m):
    if est_symetrique(m) == False:
        return False
    else:
        return True

def newnosymmetricmatrix(n):
    return np.random.randint(-100,100,(n,n))
np.complex128(1+0j)
#For an n×n symmetric matrix a, the correspondence
# with the n(n+1)/2 vector v is
# v_i(i−1)/2+j = ai,j , for i ≥ j
# In C > v[i*(i+1)/2+j] = a[i,j]
def newsymmetricmatrix(n):
    m = np.random.randint(0,100,(n,n))
    for i in range(n):
        for j in range(i,n):
            m[j,i] = m[i,j]

    return m.astype(np.float32)

def newtridiagsymmetricmatrix(n):
    return True

def newtridiagnosymmetricmatrix(n):
    return True

def new_symmetric_linearsystem(n):
    a = np.random.randint(1,100,(n,n))
    b = np.zeros(n)

    for i in range(n):
        for j in range(i,n):
            a[j][i] = a[i][j]

    for i in range(n):
        for j in range(n):
            b[i] += a[i][j]

    return a.astype(np.float32),b.astype(np.float32)

def new_symmetricpositivedefinite_linearsystem2(n: int) -> np.ndarray:
    rng = np.random.default_rng()

    M: np.ndarray = rng.integers(low=0, high=np.iinfo(np.uint32).max/2, size=(n,n), endpoint=True)
    X: np.ndarray = M.T@M
    D: np.ndarray = np.diag(rng.integers(low=np.iinfo(np.uint32.max/2, high=np.iinfo(np.ubyte).max, size=n, endpoint=True)))
    return M+D

def new_symmetricpositivedefinite_linearsystem(n: int) -> np.ndarray:
    rng = np.random.default_rng()
    m: np.float64 = rng.random(size=(n,n))
    d: np.ndarray = np.diag(rng.integers(low=10, high=100, size=n))
    p: np.ndarray = m.T@m
    a = np.zeros((n,n), dtype=np.float64)
    a += p+d**2
    x = rng.integers(low=-8, high=7, size=n, dtype=np.byte)
    b = np.zeros(n, dtype=np.float64)

    if np.all(np.linalg.eigvals(a) > 0):
        for i in range(n):
            for j in range(n):
                b[i] += a[i,j] * x[j]
#        print("a =", a)
#        print("x =", x)
#        print("b =", b)
#        return a.astype(np.float32),b.astype(np.float32)
        return a, b
    else:
        print("KO !")

def systeme_lineaire_non_symetrique(n: np.uint8):# -> tuple(a:np.ndarray, b:np.ndarray):
    rng = np.random.default_rng()
    # a = rng.random(size=(n,n))
    # x = rng.random(size=n)
    a = rng.integers(low=1, high=10, size=(n,n), dtype = np.int8)
    x = rng.integers(low=1, high=10, size=n, dtype = np.int8)
    b = np.zeros(n, dtype = np.int8)

    for i in range(n):
        a[i,i] *= 5

    for i in range(n):
        for j in range(n):
            b[i] += a[i,j] * x[j]
    
    print ("a=", a)
    print ("x=", x)
    print ("b=", b)

    if np.linalg.det(a) == 0:
        raise("|a| = 0 !")
    return a.astype(np.int8), b.astype(np.int8)

def echanger_lignes(mat,l1,l2):
    l1 -= 1
    l2 -= 1
    mat[[l1,l2],:] = mat[[l2,l1],:]

def echanger_colonnes(mat,c1,c2):
    c1 -= 1
    c2 -= 1
    mat[:,[c1,c2]] = mat[:,[c2,c1]]


def toeplitz_tridiagonale(va: np.int_, vb: np.int_, vd: np.int_, n: np.uint) -> np.ndarray:
    a: np.ndarray = np.repeat(va, n-1)
    b: np.ndarray = np.repeat(vb, n-1)
    d: np.ndarray = np.repeat(vd, n)
    return np.diag(a,-1) + np.diag(d,0) + np.diag(b,1)

def ToeplitzTriDiagNonSymEV(m: np.ndarray) -> list:
    n = m.shape[0]
    ev = []
    for i in range(n):
        ev.append(d[i] + 2 * np.sqrt(a[i]*b[i])) * np.cos((i * np.pi) / (n+1))
    return ev

# divide(a,b) = a / b
# divide(a::Integer, b::Integer) = div(a, b)

# function dodgson!(a::abstractMatrix{T}, B::abstractMatrix{T}) where {T<:Number}
#     m,n = size(a)
#     m == n > 0 || throw(argumentError("det requires a square matrix"))
#     size(B,1) ≥ m-1 && size(B,2) ≥ m-1 || throw(DimensionMismatch())
#     B .= 1
#     while m > 1
#         for j=1:m-1, i=1:m-1
#             B[i,j] = a[i,j]
#             a[i,j] = divide(a[i,j]*a[i+1,j+1] - a[i+1,j]*a[i,j+1], B[i+1,j+1])
#         end
#         m -= 1
#     end
#     return a[1,1]
# end

# dodgson(a::abstractMatrix{<:Number}) = dodgson!(Base.copymutable(a), Base.copymutable(a))

def Dodgson(mat):
    n = len(mat)
    b = np.zeros((n,n),dtype='float64')
    c = np.zeros((n,n),dtype='float64')
    while n > 0:
        for i in range(n):
            for j in range(n):
                b[i][j] = mat[i][j]
                b[i][j] = mat[i][j]*mat[i][j]-mat[i][j]*mat[i][j]
        for i in range(n):
            for j in range(n):
                b[i][j] = b[i][j] / (c[i][j])
        c = mat
        mat = b
    return n[0][0]

def bareiss(a: np.ndarray) -> np.int32:
    n = a.shape[0]
    for k in range(n-1):
        for i in range(k+1,n):
            for j in range(k+1,n):
                a[i,j] = (a[i,j]*a[k,k] - a[i,k]*a[k,j])
                if k:
                    a[i,j] = a[i,j] / (a[k-1,k-1])

    return a[n-1, n-1]

def prodvecmat(m,v):
    n = len(m)
    w = np.zeros(n)
    
    for i in range(0,n):
        wi = 0
        for j in range(0,n):
            wi += m[i][j] * v[j]
        w[i] = wi
    return w

def Cramers2x2(a: np.float128, b: np.float128) -> np.ndarray:
    x = np.zeros(2)
    det = a[0,0]*a[1,1] - a[1,0]*a[0,1]
    if np.isclose(det, 0.0, atol=1e-8) == True:
        raise ValueError("Le déterminant est nul, il n'y a pas de solution.")
    det1 = b[0]*a[1,1] - b[1]*a[0,1]
    det2 = a[0,0]*b[1] - a[1,0]*b[0]
    if (np.isclose(det1, 0.0, atol=1e-8) == True) or (np.isclose(det2, 0.0, atol=1e-8)):
        raise ValueError("Ce système n'a aucune solution ou bien une infinité.")
    x[0] = det1 / det
    x[1] = det2 / det
    return x

def Sarrus3x3(m):
    return m[0,0]*m[1,1]*m[2,2]+m[0,1]*m[1,2]*m[2,0]+m[0,2]*m[1,0]*m[2,1] \
            -m[2,0]*m[1,1]*m[0,2]-m[2,1]*m[1,2]*m[0,0]-m[2,2]*m[1,0]*m[0,1]

def Inverse3x3(m: np.ndarray) -> np.ndarray:
    det = Sarrus3x3(m)
    if np.isclose(det, 0.0, atol=1e-8) == True:
        raise ValueError("Cette matrice n'est pas inversible.")
        os.exit(2)

    inv = np.zeros((3,3))
    inv[0,0] = (m[1,1]*m[2,2] - m[2,1]*m[1,2]) / det
    inv[0,1] = (-(m[1,0]*m[2,2] - m[2,0]*m[1,2])) / det
    inv[0,2] = (m[1,0]*m[2,1] - m[2,0]*m[1,1]) / det
    inv[1,0] = (-(m[0,1]*m[2,2] - m[2,1]*m[0,2])) / det
    inv[1,1] = (m[0,0]*m[2,2] - m[2,0]*m[0,2]) / det
    inv[1,2] = (-(m[0,0]*m[2,1] - m[2,0]*m[0,1])) / det
    inv[2,0] = (m[0,1]*m[1,2] - m[1,1]*m[0,2]) / det
    inv[2,1] = (-(m[0,0]*m[1,2] - m[1,0]*m[0,2])) / det
    inv[2,2] = (m[0,0]*m[1,1] - m[1,0]*m[0,1]) / det

    return inv.T

# def addmatrix(a,b):
#     n = len(a)
#     if n != len(b):
#         print("Les matrices n'ont pas la même taille.")
#     c = np.zeros((n,n),dtype='int')
#     for i in range(n): 
#         c = np.append([sum(pair) for pair in zip(a[i], b[i])]) 

#     return c
# list1 = [11, 21, 34, 12, 31, 77] 
# list2 = [23, 25, 54, 24, 20] 
# result = [sum(i) for i in zip(list1, list2)]  

def mattriupproduct(a,b):
    n = len(a)
    if (n != len(b)):
        print('error dim(a) != b\n')

    c = np.zeros((n,n),dtype='float64')
    for i in range(1,n):
        for j in range(i-1,n):
            for k in range(i-1,j):
                c[i][j] += a[i][k]*b[k][j]
    return c

def prodmat1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    if (n != b.shape[0]):
        print("Ordres incompatibles !")
    c = np.zeros((n,n),dtype='float64')
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[i][j] += a[i][k]*b[k][j]
    return c

def prodmat2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    c = np.zeros((n,n))
    for i in range(n):
        c[i] = np.dot(a[i],b)
    return c

def prodmat3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    if (n != b.shape[0]):
        print("Ordres incompatibles !")
    c = np.zeros((n,n),dtype='float64')
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[i * n  + k] += a[i * n + j] * b[j * n + k];
    return c

# def ProduitMatriciel(a:np.ndarray, b:np.ndarray) -> np.ndarray:
#     n = a.shape[0]
#     r = b.shape[0]
#     s = b.shape[1]
#     if n == s:
#         c = np.zeros((n,s))
#         for i in range(n):
#             for j in range(s):
#                 for k in range(r):
#                     c[i,j] = a[i,k] * b[k,j]
#         return c
#     else:
#         print("Erreur.")

# >>> a = np.array([[-3,5],[1,6],[-1,2]])
# >>> b = np.array([[1,7,3],[-5,-2,4]])
# >>> ProduitMatriciel(a,b)

    # a11=np.array(a[:m,:m], [m,m])
    # a12=np.array(a[m+1:,:m], [m,m])
    # a21=np.array(a[:m,:m+1], [m,m])
    # a22=np.array(a[:m+1,:m+1], [m,m])
    # b11=np.array(b[:m,:m], [m,m])
    # b12=np.array(b[m+1:,:m], [m,m])
    # b21=np.array(b[:m,:m+1], [m,m])
    # b22=np.array(b[:m+1,:m+1], [m,m])

def blockmatproduct(a: np.ndarray, b: np.ndarray, s: np.int16) -> np.ndarray:
    n = len(a)
#    if (n != len(b)):
#        print('error dim(a) != b\n')
#
#    if (n%s != 0):
#        print('s n\'est pas un multiple de n')
#        quit()
    if (n != len(b)) or (n%s != 0):
        print('a != b or n neq s')
        quit()

    c = np.zeros((n,n),dtype='float64')

    for i in range(int(n/s)):
        for j in range(int(n/s)):
            for k in range(n):
                for si in range(s):
                    for sj in range(s):
                        c[i*s+si][j*s+sj] += a[i*s+si][k] * b[k][j*s+sj]

    return c

def hadamardproduit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    c = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            c[i,j] = a[i,j] * b[i,j]
    return c

def kroneckerproduit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r = a.shape[0]
    s = a.shape[1]
    t = b.shape[0]
    u = b.shape[1]
    c = np.zeros((r*t,s*u))
    for i in range(0,r):
        for j in range(0,s):
            for k in range(0,t):
                for l in range(0,u):
                     c[k+i*t,l+j*u] = a[i,j] * b[k,l]
    return c

def kroneckersomme(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ia = np.identity(a.shape[0])
    ib = np.identity(b.shape[0])
    return kroneckerproduit(a,ib) + kroneckerproduit(ia,b)

def khatriraoproduit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = a.shape[1]
    tmp = np.zeros((n,2*n+1))
    for i in range(b.shape[1]):
        tmp = map(kroneckerproduit(a[:,i], b[:,i]))
    return c

def strassen(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(a)
    if (n != len(b) and n % 2):
        raise Exception("Matrices d'ordre pair uniquement !")

    def divm(m):
        n = len(m)
        nd2 = n//2
        return (m[:nd2, :nd2], m[:nd2, :nd2], m[nd2:, :nd2], m[nd2:, :nd2])

    m = int(n/2)
    a11: np.ndarray = np.mat(a[:m,:m])
    a12: np.ndarray = np.mat(a[:m,m:])
    a21: np.ndarray = np.mat(a[m:,:m])
    a22: np.ndarray = np.mat(a[m:,m:])
    b11: np.ndarray = np.mat(a[:m,:m])
    b12: np.ndarray = np.mat(a[:m,m:])
    b21: np.ndarray = np.mat(a[m:,:m])
    b22: np.ndarray = np.mat(a[m:,m:])

    p1: np.ndarray = strassen(a11+a22, b11+b22)
    p2: np.ndarray = strassen(a21+a22, b11)
    p3: np.ndarray = strassen(a11, b12-b22)
    p4: np.ndarray = strassen(a22, b21-b11)
    p5: np.ndarray = strassen(a11+a12, b22)
    p6: np.ndarray = strassen(a21-a11, b11+b12)
    p7: np.ndarray = strassen(a12-a22, b21+b22)

    c = np.zeros((n,n), dtype = a.dtype)
    c[:m,:m] = p1+p4-p5+p7
    c[m+1:,:m] = p2+p4
    c[:m,:m+1] = p3+p5
    c[:m+1,:n/2+1] = p1+p3-p2+p6

    return np.matrix([[c[:m,:m], c[m+1:,:m]], [c[:m,:m+1], c[:m+1,:n/2+1]]])

def faddeev_leverier(a: np.ndarray) -> np.polynomial.Polynomial:
    n = a.shape[0]
    coefs = []
    ak = np.copy(a)
    i = np.identity(n)

    for k in range(n+1):
        c = (1.0/(k+1)) * np.trace(ak)
        coefs.append(c)
        ak = a @ (ak - c*i)
        if np.all(ak == 0):
            break
    coefs.insert(0,1)
    np.polynomial.set_default_printstyle('unicode')
    return np.polynomial.Polynomial(coefs[::-1])

def Substitutionavant(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    x = np.array(b)
    for i in range(n):
        x[i] = b[i]
        for j in range(i):
            x[i] -= a[i,j] * x[j]
        x[i] /= a[i,i]

    return x

def Substitutionarriere(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    x = np.array(b)
    for i in range(n-1,-1,-1):
        s = 0
        for j in range(i+1,n):
            s += a[i][j] * x[j]
        x[i] = (x[i] - s) / a[i][i]

    return x

# public class Main {
#     public static double[][] invertUpperTriangular(double[][] r) {
#         int dim = Math.min(r.length, r[0].length);
#         double[][] inv = new double[dim][dim];

#         for (int i = dim - 1; i >= 0; i--) {
#             inv[i][i] = 1.0 / r[i][i];
#             for (int j = i + 1; j < dim; j++) {
#                 for (int k = i + 1; k < dim; k++) {
#                     inv[i][j] -= r[i][k] * inv[k][j];
#                 }
#                 inv[i][j] *= inv[i][i];
#             }
#         }
#         return inv;
#     }
# }

#    public void invertUpper (double u[][], int n)
#                     throws NotFullRankException {
#       double sum;
#       int i,j,k;
#       int err;
#       err = 0;
#       for (i = 0; i < n; i++) {
#          if (u[i][i] == 0.0) {
#             System.out.println("\nTriangular.invertUpper error:" +
#             "  Diagonal element " + i +
#             " of the U matrix is zero.\n");
#             err = 1;
#          }
#       }

#       if (err == 1) {
#             throw new NotFullRankException();
#       }
#       for (j = n - 1; j > -1; j--) {
#          u[j][j] = 1.0/u[j][j];
#          for (i = j - 1; i > -1; i--) {
#             sum = 0.0;
#             for (k = j; k > i; k--) {
#                sum -= u[i][k]*u[k][j];
#             }
#             u[i][j] = sum/u[i][i];
#          }
#       }
#       return;
#    }

def inv_tri_sup(a: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    inv = np.zeros((n,n))
    for i in range(n):
        if (a[i][i] == 0.0):
            raise np.linalg.LinalgError("Déterminant nul !")
        else:
            inv[i,i] = 1.0 / a[i,i]
        for j in range(i):
            for k in range(j,i):
                inv[j,i] += a[k,i] * inv[j,k]
            inv[j,i] = -inv[j,i] / a[i,i]
    return inv

def inv_tri_inf(a: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    inv = np.zeros((n,n))
    for i in range(n):
        if (a[i][i] == 0.0):
            raise np.linalg.LinalgError("Déterminant nul !")
        else:
            inv[i,i] = 1.0 / a[i,i]
        for j in range(i):
            for k in range(j,i):
                inv[i,j] += a[i,k] * inv[k,j]
            inv[i,j] = -inv[i,j] / a[i,i]
    return inv

def pivotgauss(
    a: np.ndarray,
    b: np.ndarray,
    k: int = 20,
    tol: float = 1e-8
    ) -> np.ndarray:

    n = a.shape[0]
    if (n != b.shape[0]):
        print("Ordre de a != ordre de b.")
        exit()
    for i in range(n):
        pivot = a[i,i]
        # if np.less(pivot, 0.0) == True:
        if (pivot == 0.0 or np.fabs(pivot) <= tol):
            max_row = i
            for j in range(i+1,n):
                if (np.fabs(a[j,i]) > np.fabs(a[max_row,i])):
                    max_row = j
            for k in range(i,n+1):
                tmp = a[i,k-1]
                a[i,k-1] = a[max_row,k-1]
                a[max_row,k-1] = tmp
        else:
            for j in range(i+1,n):
                tmp = a[j,i] / pivot
                for k in range(i+1,n):
                    a[j,k] = a[j,k] - tmp * a[i,k]
                    a[j,i] = 0.0
                b[j] -= tmp * b[i]
    x = np.copy(b)
    for i in range(n-1,-1,-1):
        for j in range(i+1,n):
            x[i] -= a[i,j] * x[j]
        x[i] /= a[i,i]

    return x

def gauss_bareiss(a, b):
    n = len(a)
    if (n != len(b)):
        print('Dim(a) != dim(b).\n')

    x = np.zeros(n)

    return x

# For k from 0 to p-1 do 
#     b(k+1) = b(k)/a(k) k kk,k
#     For j from k+1 to p-1 do
#         a(k,j) = a(k,j)/a(k,k)
#         For i from 0 to p-1 do
#             If k̸=i then
#             a(i,j) = a(i,j) - a(i,k) * a(k,j)^(k+1)
#             End if
#         End for
#     End for
#     For i from 0 to p-1 do
#         If k̸=i then
#             b(i) = b(i) - a(i,k) * b(k)^(k+1)
#         End if
#     End for
# End for

def GaussJordan(
    m: np.ndarray,
    ) -> np.ndarray:

    d = np.linalg.det(a) 
    if d == 0.0:
        print("det=0")
        exit()

    n = m.shape[0]
    inv = np.eye(n, n*2, 0)

    for i in range(n):
        for k in range(n):
            kk = m[k,k]
            ikk = 1/kk
            for j in range(2*n):
                inv[k,j] *= ikk

    for i in range(n):
        if i == k:
            continue
        mik = m[i,k]
        for j in range(2*n):
            inv[i,j] -= inv[k,j] * mik
    
    # if np.close(m@inv, np.identity(n)) == True:
    #     return inv
    # else:
    #     raise arithmeticError("Inverse : non !")
    return inv

def lu(
    a: np.ndarray,
    b: np.ndarray,
    k: int = 20,
    tol: float = 1e-8
    ) -> np.ndarray:

    n = a.shape[0]
    if (n != b.shape[0]):
        print("Ordre de a != ordre de b.")
        exit()

    l = np.identity(n)
    u = np.zeros((n,n))
    x = np.zeros(n)
    y = np.zeros(n)

    for i in range(n):
        for k in range(i):
            l[i,k] = a[i,k]
            for j in range(k):
                l[i,k] -= l[i,j]*u[j,k]
            l[i,k] /= u[k,k]

        for k in range(i,n):
            u[i,k] = a[i,k]
            for j in range(i):
                u[i,k] -= l[i,j]*u[j,k]

    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= l[i,j]*y[j]

    for i in range(n-1,-1,-1):
        x[i] = y[i]
        for j in range(i+1,n):
            x[i] -= u[i,j]*x[j]
        x[i] /= u[i,i]

    return x

def lu_factorisation(a):
    n = len(a)
    l = np.identity(n)
    u = np.zeros((n,n),dtype='float64')

    for i in range(n):
        for k in range(i):
            l[i][k] = a[i][k]
            for j in range(k):
                l[i][k] -= l[i][j]*u[j][k]
            l[i][k] /= u[k][k]

        for k in range(i,n):
            u[i][k] = a[i][k]
            for j in range(i):
                u[i][k] -= l[i][j]*u[j][k]
    return l, u

def cholesky(
    l: np.ndarray,
    b: np.ndarray,
    k: int = 50,
    tol: float = 1e-8
    ) -> np.ndarray:
    n = l.shape[0]

    for i in range(n):
        d = l[i,i]
        for j in range(n):
            if l[i,j] != l[j,i] or d <= 0:
                print('La matrice n\'est pas symétrique et définie positive.\n')
                return False

    for j in range(n):
        for k in range(j):
            l[j,j] -= l[j,k] * l[j,k]
        tmp = l[j,j]
        print(tmp)
        l[j,j] = np.sqrt(tmp)
        for i in range(j+1,n):
            l[j,i] = 0
            for k in range(j-1):
                l[i,j] -= l[j,k] * l[i,k]
            l[i,j] = l[i,j] / l[j,j]

    print(l)
    x = np.zeros(n)
    y = np.zeros(n)

    # Ly = b
    y[0] = b[0] / l[0][0]
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += l[i,j] * y[j]
        y[i] = (b[i] - s) / l[i,i]

    # Ux = y
    lt = np.transpose(l)
    x[n-1] = y[n-1] / lt[n-1,n-1]
    s = 0.0
    for i in range(n-1,-1,-1):
        s = y[i]
        for j in range(i+1,n):
            s -= lt[i,j]*x[j]
        x[i] = s / lt[i,i]

    return x

# ChI=tril(a)
# for k=1:nn
#     ChI(k,k)=np.sqrt(ChI(k,k));
#     for j=k+1:nn
#         if ChI(j,k) ~= 0
#             ChI(j,k)=ChI(j,k)/ChI(k,k);
#         end
#     end
#     for j=k+1:nn
#         for i=j:n
#             if ChI(i,j) ~= 0 ChI(i,j)=ChI(i,j)-ChI(i,k)-ChI(j,k);
#             end
#         end
#     end
# end

#def lutri(c,d,e):
#    n = len(d)
#
#    u = np.array(d, copy=True)
#    l = np.array(c, copy=True)
#
#    for i in range(1,n+1):
#        l[i] = c[i]/u[i]
#        u[i] = d[i]-l[i+1]*e[i]
#
#    return l, u

# def tri():
#     n = 5
#     m = np.zeros((n,n))
#     for i in range(n):
#         m[i, i] = 2 # diagonale
#         if i < n:
#             m[i-1, i] = 1 # sur-diagonale
#             print("m[i-1, i]=", m[i-1, i])
#         if i >= 1:
#             m[i, i-1] = 3 # sous-diagonale
#     return m

def thomas(
    c: np.ndarray,
    d: np.ndarray,
    e: np.ndarray,
    b: np.ndarray
    ) -> np.ndarray:
    n = d.shape[0]

    r = np.zeros(n)
    s = np.zeros(n)
    x = np.zeros(n)

    r[0] = d[0]
    s[0] = b[0]/r[0]

    for i in range(1,n):
        r[i] = d[i] - (c[i]*e[i-1]/r[i-1])
        s[i] = (b[i]-c[i]*s[i-1])/r[i]

    x[n-1] = s[n-1]
    for i in range(n-2,-1,-1):
        x[i] = s[i] - e[i] * x[i+1] / r[i]

    return x

# d g h 0 0 0 ...
# f d g h 0 0 ...
# e f d g h 0 ...
# 0 e f d g h ...
# 0 0 e f d g h
# 0 0 0 e f d g h
# mettre en oeuvre https://arxiv.org/pdf/1409.4802.pdf
def pentad(e,f,d,g,h,b):
    n = d.shape[0]
    if (n != b.shape[0]):
        print("Ordre de a != ordre de b.")
        exit()

    x = np.zeros(n)

    for i in range(2,n-1):
        tmp1 = f[i-1]/d[i-1]
        d[i] -= tmp1*g[i-1]
        g[i] -= tmp1*h[i-1]
        b[i] -= tmp1*b[i-1]
        tmp2 = e[i-1]/d[i-1]
        f[i] -= tmp2*g[i-1]
        d[i+1] -= tmp2*h[i-1]
        b[i+1] -= tmp2*b[i-1]
    
    tmp3 = f[n-1]/d[n-1]
    print('d[n-1]=', d[n-1])
    print('b[n-1]=', b[n-1])
    d[n-1] -= tmp3 * g[i]
    x[n] = b[n] - tmp3*b[n-1]/d[n]
    x[n-1] = b[n-1] - g[n-1]*b[n]/d[n-1]
    for i in range(n-2,1):
        x[i] = b[i] - g[i]*x[i+1]-h[i+2]/d[i]

    return x

def gauss_seydel(
    a: np.ndarray,
    b: np.ndarray,
    k: int = 50,
    tol: np.float128 = 1e-15
    ) -> np.ndarray:

    n = a.shape[0]
    x0 = np.ones_like(b)
    x1 = np.ones_like(b)

    for _ in range(k):
        for i in range(n):
            s = 0
            for j in range(n):
                if (j != i):
                    s += a[i,j] * x0[j]
                x1[i] = (b[i] - s) / a[i,i]
        norme = np.linalg.norm(b - a @ x1, ord=np.inf)
        if np.less(norme, tol) == True:
            break
        else:
            x0 = x1

    return x1


def sor(
    a: np.ndarray,
    b: np.ndarray,
    omega: float,
    k: int = 100,
    tol: float = np.finfo(np.float64).eps
    ) -> np.ndarray:

    n = a.shape[0]
    x0 = np.ones_like(b, dtype=np.float64)
    x1 = np.ones_like(b, dtype=np.float64)
    s:np.float64 = 0

    it: int = 1
    while it < k:
        for i in range(n):
            s = 0
            for j in range(i):
                if (j != i):
                    s += a[i,j] * x1[j]
            for j in range(i,n):
                if (j != i):
                    s += a[i,j] * x0[j]
            x1[i] = (1 - omega) * x0[i] + (omega * (b[i] - s)) / a[i,i]

        norme = np.linalg.norm(b - a @ x1)
        if np.less(norme, tol) == True or it <= k:
            break
        else:
            x0 = x1
            it += 1

    return x1.astype(np.float64)


# https://www.joezhouman.com/2021/11/21/NumericalanalysisIteration.html
def aor(
    a: np.ndarray,
    b: np.ndarray,
    omega: float,
    r: float,
    k: int,
    tol: np.float128 = 1e-15
    ) -> np.ndarray:

    n = a.shape[0]
    x0 = x1 = np.ones_like(b)
    s = np.zeros(3)

    for _ in range(k):
        for i in range(n):
            s[0] = s[1] = s[2] = 0
            for j in range(i):
                aij = a[i,j]
                s[0] += aij * x1[j]
                s[1] += aij * x0[j]
            for j in range(i+1,n):
                s[2] += a[i,j] * x0[j]
            aii = a[i,i]
            x1[i] = x0[i] + omega * ((b[i] - s[1] - s[2]) / aii - x0[i]) + r * (s[1] - s[0]) / aii
        norme = np.linalg.norm(b - a @ x1, ord=np.inf)
        if np.less(norme, tol) == True:
            break
        else:
            x0 = x1

    return x1


# def aor(a, b, w=1.0, r=1.0, tol=1e-8, max_iter=1000):
#     # L = D^(-1)a_L
#     # U = D^(-1)a_U
#     # c = D^(-1)b
#     # (I-rL)x^(n+1) = [(1-w)I + (w-r)L + wU]x^(n) + wc

#     n = len(b)
#     xt = np.zeros(n)
#     x0 = np.ones(n)
#     x1 = np.ones(n)
#     I = np.eye(n)
#     aL = np.tril(a, k=-1)
#     aU = np.triu(a, k=1)
#     D = np.diag(np.diagonal(a))

#     L:np.ndarray = (1 / D) * (-aL)
#     U:np.ndarray = (1 / D) * (-aU)
#     c:np.ndarray = (1 / D) * b

#     for _ in range(k):
#         xt = (I * (1-w) + L * (w-r) + w * U)
#         x1 = (xt * x0 + w * c) / (I - r * L)
#         if np.linalg.norm(x1 - x0, ord=np.inf) < tol:
#             break
#         else:
#             x0 = x1

#     return x1

# def extractL(a:np.ndarray) -> np.ndarray:
#     n = a.shape[0]
#     l = np.zeros((n,n))
#     for i in range(n):
#         for j in range(i):
#             for k in range(j,i):
#                 l[i,j] = a[j,k]


# def BlockGradientConjugue1(a,b,k,err):
#     r^{0} = b − aX^{0}
#     p^{0} = M^{−1}r^{0}
#     rho^{0} = ⟨p^{0},r^{0}⟩
#     for k = 0,...
#         Q^{k} = aP^{k}
#         alpha^{k} = ⟨Pk,Qk⟩
#         lambda^{k} = (alpha^k)^{-1} * rho^{k}
#         X^{k+1} = X^{k} + P^{k} * λ^{k}
#         R^{k+1} = R^{k} − Q^{k} * lambda^{k}
#         break if |Rk+1| < eps
#         Z^{k+1} = M^{−1} * R^{k+1}
#         rho^{k+1} = ⟨Z^{k+1}, R^{k+1}⟩
#         beta^{k} = (rho^{k})^{−1} * rho^{k+1}
#         P^{k+1} = Z^{k+1} + P^{k} * beta^{k}

#https://files.core.ac.uk/pdf/23/81673055.pdf

# def prodvecmat_coo(m,v):
# def prodvecmat_csr(m,v):
# def prodmatmat_coo(m,v):
# def prodmatmat_csr(m,v):

# algorithm: SpMV on CSR format
# Input: SSBO a, Ia, Ja, P, Row Output: SSBO Result
# 1: i → gl_GlobalInvocationID.x
# 2: if i < Row then
# 3 value→0
# 3: rowStart → Ia[i]
# 4: rowEnd → Ia[i+1]
# 5: for j = rowStart to rowEnd do
# 6: value → value + a[j] * P[Ja[j]]
# 7: end for
# 8: Result[i] → value
# 9: end if

# M Number of rows in matrix
# N Number of columns in matrix
# NZ Number of nonzeros in matrix MaXNZR Maximum number of nonzeros per row NDIaG Numero of nonzero diagonals
# aS Coefficients array
# Ia Row Indices array
# Ja Column Indices array
# IRP Row Start Pointers array
# NZR Number of Nonzeros per row array OFFSET Offset for diagonals

# #pragma omp parallel for
# SpMV CSR
# for (int i=0; i<n; ++i) {
#     y[i] = 0.0;
#     for (int j=row_off[i]; j<row_off[i+1]; ++j)
#         y[i] += val[j]*x[col[j]];
# }

# Matrix-Vector product in CSR format
# CSR
# for i = 1:m
# t=0
#     for j = irp(i); irp(i+)−1; j++
#       t = t + as(j) * x(ja(j))
#     end
#     y(i)= t
# end

# CSR
# CSR_SPMV_T(a, x, y)
# n=a.cols
# for i=0 to n-1
#   y[i]=0
# for i = 0 to n - 1
#   for k from a. row_ptr[i] to a. row_ptr[i + 1] - 1
#       y[a. col_ind[k]] = y[a. col_ind[k]] + a. val[k] * x[i]

# CSR
# input: a : sparse n × n matrix,
# v : dense vector of length n.
# output: u : dense vector of length n, u = av.
#
# for i = 0 to n - 1 do
#     u[i] = 0;
#     for k = start[i] to start[i + 1] − 1
#         u[i] = u[i] + a[k] · v [j[k]]

# CSR the sparse matrix-vector product.
# // y = a*x
# y = zeros(row.size() - 1)
# for (i = 0; i < row.size()-1; i++)
#   for (j = row[i]; j < row[i+1]; j++)
#     // Parcours des indices colonnes de la ligne i
#     y[i] += val[j]*x[col[j]];
#   end
# end

# CSR the sparse matrix-vector product.
# for (row=0; row<nrows; row++) {
#     s = 0;
#     for (icol=ptr[row]; icol<ptr[row+1]; icol++) {
#         int col = ind[icol];
#         s += a[icol] * x[col];
#     }
#     y[row] = s;
# }

# CSR the sparse matrix-vector product.
# for i = 0 to dimRow-1 do
#     row = rowP trs[i]
#     y[i] = 0
#     for j = 0 to rowPtrs[i+1]-row-1 do
#         y[i]+= values[row+j]∗x[colIdxs[row+j]]
#     end for
# end for

# Matrix-Vector product in CSR format y = ax
# n > ordre
# nnz > nombre de valeurs non nulles
# rows_idx > position de la première entrée de chaque ligne
# colnum > numéro de la colonne de a
# coef > valeur non nulle
# x > vecteur
# sortie > y vecteur
# i, j, idx > variables
# for i=0 to n-1
#     y[i]=0
# for i=0 to n-1
#     for idx = rows_idx[i] to rows_idx[i+1] - 1
#         j = colnum[idx]
#         y[i] = y[i] + coef[idx] * x[j]
#     end
# end
 
# algorithm 3.2: Matrix-vector multiplica- tion with the COO storage format
# input : n (size of the matrix), nnz (number of non-zero values),
#         r_numb (row numbers),
#         c_numb (column numbers),
#         coef (non-zero values),
#         x (vector)
# output: y (vector)
# variable: i, j, tid
# // - - initialization of the vector result
# for i=0 to n−1 do y[i]←0
# // - - compute the COO SpMV
# for tid=0 to nnz−1 do
#     i ← r_numb[tid]
#     j ← c_numb[tid]
#     y[i] ← y[i] + coef[tid] × x[j]
# end

# algorithm 3.4: Matrix-vector multiplication with the CSR storage format
# input : n (size of the matrix),
#           nnz (number of non-zero values),
#           rows_indices (position of the first entry of each row),
#           c_numb (matrix column numbers), coef (matrix non-zero values),
#           x (vector)
# output : y (vector) variable : i, j, tid
# // - - initialization of the vector result
# for i=0 to n−1 do
#   y[i]←0
# for i=0 to n−1 do
#     for tid = rows_indices[i] to rows_indices[i + 1] − 1 do
#         j ← c_numb[tid]
#         y[i] ← y[i] + coef[tid] * x[j]
#     end
# end

# Sparse matrix vector product for the compressed sparse row (CSR) format.
# for i = 0 to n − 1 do
#     y[i]←0
#     for j = row_pointer[i] to row_pointer[i + 1] − 1 do
#         y[i] ← y[i] + values[ j] * x[column_index[j]]
#     end
# end

# Sparse matrix vector product for the compressed sparse row (CSR) format.
# void spmxv(const double *val, const int *col,
#            const int *row, const double *x, double *v, int nrows)
# {
# register int i,j;
# register double d0;
#
# #pragma omp for private(i,j,d0) nowait
# for (i = 0; i < nrows; i++) {
#     d0 = 0.0;
#     for (j = row[i]; j < row[i+1]; j++)
#         d0 += val[j] * x[col[j]];
#         v[i] = d0;
#     }
# }

def ProduitMatriceVecteurCOO(v, n, ligne, val , col):
    if (n != len(v)):
        print('Error: dim(M) != dim(v)')
        return 0
    v = np.zeros(n)

    for i in range(n):
        v[ligne[i]] += val[i]*x[col[i]]
        
    return v

def ProduitMatriceVecteurCSR(v, n ,ligne, valeur, col):
    if (n != len(v)):
        print('Error: dim(M) != dim(v)')
        return 0
    v = np.zeros(n)

    for i in range(n):
        for j in range(ligne[i], j < ligne[i+1]):
            v[i] += valeur[j]*x[col[j]]

    return v

def ProduitScalaireCreux(a, b):
    n = len(a)
    result = 0
    i = j = 0
    while (i < n and j < n):
        if (a[i][0] == b[j][0]):
            result += a[i][1] * b[j][1]
            i += 1
            j += 1
        elif a[i][0] > b[j][0]:
            i += 1
        else:
            j += 1
    return result

# procedure BLOCKCG
# R(0) = B-aX(0)
# P(0) = R(0)
# for k = 1,2,... until converged do
#     Z(k-1) =aP(k-1)
#     alpha(k-1) = (P(k-1))^H Z(k-1)^(-1) (R(k-1))^H R(k-1)
#     X(k) = X(k-1) + P(k-1) alpha(k-1)
#     R(k) = R(k-1) - Z(k-1) alpha(k-1)
#     beta(k-1) = (R(k-1))^H R(k-1)^(-1) (R(k))^H R(k)
#     P(k) = R(k) - P(k-1) beta(k-1)
# end for
# end procedure

# def LireMatrixMarket(chemin):
#     if os.path.isdir(chemin) == False:
#         print("Ce dossier n'existe pas.")
#         exit(0)
#     liste = os.listdir(chemin)
#     del liste[0] # .DS_Store !
#     matrice = chemin+liste[0]
#     vecteur = chemin+liste[1]
#     print("Fichiers à charger :")
#     print("--------------------\n")
#     print("Matrice :", matrice)
#     print(spio.mminfo(matrice))
#     print("\n")
#     print("Vecteur :", vecteur)
#     print(spio.mminfo(vecteur))
#     print("\n")

#     print("Lecture des fichiers... ", end="")
#     acoo = spio.mmread(matrice).astype(np.float32)
#     bcoo = spio.mmread(vecteur).astype(np.float32)
#     print("ok")

#     print("acoo : ", acoo)
#     print("bcoo : ", bcoo.T)
#     return acoo, bcoo

import scipy.io
import scipy.sparse

def GradientConjugue(
    a: np.ndarray,
    b: np.ndarray,
    ) -> np.ndarray:
    # assert a.shape[0] == b.shape[0]
    n = a.shape[0]
    # if (n != b.shape[0]):
    #     print("Ordre de a != ordre de b.")
    #     exit()

    for i in range(n):
        for j in range(n):
            if a[i,j] != a[j,i] or a[i,i] <= 0:
                print("La matrice a n'est pas SPD.")
                return False
    
    if np.allclose(a, a.T) is False:
        exit()

    x = np.ones(n)
    r = b - a @ x
    d = np.copy(r)
    rho0 = np.dot(r,r)
    tol: np.float128 = np.finfo(np.float64).resolution

    for _ in range(n):
        ad = np.dot(a,d)
        add = np.dot(ad,d)
        alpha = rho0 / add
        x += alpha * d
        r -= alpha * ad
        rho1 = np.dot(r,r)
        if np.allclose(np.linalg.norm(r, ord=np.inf), np.zeros_like(b), atol=tol) == True:
            break
        else:
            beta = rho1 / rho0
            d = r + d * beta
            rho0 = rho1

    return x

def GradientConjuguePreconditionneJacobi(
    a: np.ndarray,
    b: np.ndarray,
    ) -> np.ndarray:

    n = a.shape[0]

    D = np. diag(np.diagonal(a))
    x = np.ones_like(b)
    r0 = b - a@x
    rn = np.copy(r0)
    z0 = np.copy(r0)
    zn = np.copy(z0)
    d = np.copy(z0)
    rho0 = np.dot(r0,z0)
    tol = np.finfo(np.float64).resolution
    z0 = D @ rn
    
    for _ in range(n):
        rho0 = np.inner(r0,z0)
        ad = np.inner(a,d)
        alpha = rho0 / (np.inner(ad,d))
        x = x + alpha * d
        rn = r0 - alpha * ad
        zn = np.linalg.inv(D) @ rn
        rho1 = np.inner(rn,zn)
        beta = rho1 / rho0
        if k == 0:
            d = zn
        else:
            d = zn + beta * d
        if np.allclose(np.linalg.norm(r, ord=np.inf), np.zeros_like(b), atol=tol) == True:
            break
        else:
            rho0 = rho1
            r0 = rn
            z0 = zn

    return x


# https://github.com/styvon/cPCG/blob/master/src/rcpparma_cPCG.cpp
# https://github.com/pymatting/pymatting/blob/master/pymatting/preconditioner/ichol.py
# https://ftp.mcs.anl.gov/pub/tech_reports/reports/P682.pdf
# https://ttu-ir.tdl.org/bitstream/handle/2346/47490/KENNEDY-THESIS.pdf

# def ssor_preconditioner(a, omega):
#     n = a.shape[0]
#     M = np.zeros_like(a)
#     D = np.diag(np.diag(a))
#     L = np.tril(a, -1)
#     U = np.triu(a, 1)

#     # Compute M = (D + omega * L) * inv(D) * (D + omega * U)
#     DL_inv = np.linalg.inv(D + omega * L)
#     DU_inv = np.linalg.inv(D + omega * U)
#     M = np.dot(DL_inv, D) @ DU_inv

#     return M

def GradientConjuguePreconditionneSSOR(
    a: np.ndarray,
    b: np.ndarray,
    omega: float,
    ) -> np.ndarray:

    n = a.shape[0]

    # SSOR
    if omega < 0 or omega > 2 or None:
        omega = 1
    L = np.tril(a)
    D = np. diag(np.diagonal(a))
    # P = (1.0 / (omega * (2.0 - omega))) * (D + L * omega) @ np.linalg.inv(D) @ (D + L.T * omega)
    # M = np.linalg.inv(P.T) @ np.linalg.inv(P)

    # 1 / (2 - w) * (1/w * D + L) * inverse(1/w * D) * transpose(1/w * D + L)
    invom = 1 / omega
    dio = D * invom
    dol = dio + L
    P = dol * (1 / (2.0 - omega)) * np.linalg.inv(dio) * dol.T

    x = np.ones_like(b)
    r0 = b - a@x
    rn = np.copy(r0)
    z0 = np.copy(r0)
    zn = np.copy(z0)
    d = np.copy(z0)
    rho0 = np.dot(r0,z0)
    tol = np.finfo(np.float64).resolution
    z0 = P @ rn
    
    for _ in range(n):
        rho0 = np.inner(r0,z0)
        ad = np.inner(a,d)
        alpha = rho0 / (np.inner(ad,d))
        x = x + alpha * d
        rn = r0 - alpha * ad
        zn = P @ rn
        rho1 = np.inner(rn,zn)
        beta = rho1 / rho0
        if k == 0:
            d = zn
        else:
            d = zn + beta * d
        # if np.allclose(np.linalg.norm(r, ord=np.inf), np.zeros_like(b), atol=tol) == True:
        if np.sqrt(rho1) <= tol:
            break
        else:
            rho0 = rho1
            r0 = rn
            z0 = zn

    return x


# def GradientConjuguePreconditionneIC(
#     a: np.ndarray,
#     b: np.ndarray,
#     ) -> np.ndarray:

#     n = a.shape[0]
#     L = np.copy(a)

#     # Incomplete Cholesky Factorization With Limited Memory (Lin-Moré)
#     for j in range(n):
#         L[j,j] = np.sqrt(L[j,j])
#         for k in range(j):
#             if L[j,k] != 0:
#                 for i in range(j+1, n):
#                     if L[i,k] != 0:
#                         L[i,j] = L[i,j] - L[i,k] * L[j,k]

#         for i in range(j+1,n):
#             if L[i,j] != 0:
#                 L[i,j] = L[i,j] / L[j,j]
#                 L[i,i] = L[i,i] - L[i,j] * L[i,j]

#         for i in range(j+1,n):
#             L[j,i] = 0

#     # c  = http://sparse.tamu.edu/Nasa/nasa2146

#     x = np.ones_like(b)
#     ro = b - a@x
#     rn = np.copy(ro)
#     M = np.linalg.inv(L)
#     zo = np.copy(ro)
#     zn = np.copy(ro)
#     p = np.copy(zo)
#     ap = np.zeros_like(b)
#     app = np.zeros_like(b)
#     tol = np.finfo(np.float64).resolution

#     for _ in range(n):
#         ap = a @ p
#         app = ap @ p
#         zo = M @ ro
#         rho0 = ro @ zo
#         alpha = rho0 / app
#         rn = ro - alpha * ap
#         zn = M @ rn
#         rho1 = rn @ zn
#         beta = rho1 / rho0
#         p = zn + beta * p
#         x = x + alpha * p
#         if np.linalg.norm(rn, ord=np.inf) <= tol:
#             break
#         else:
#             rho0 = rho1
#             ro = rn
#             zo = zn

#     return x

def GradientConjuguePreconditionneIC1(
    a: np.ndarray,
    b: np.ndarray,
    ) -> np.ndarray:

    n = a.shape[0]
    L = np.copy(a)

    # Incomplete Cholesky Factorization With Limited Memory (Lin-Moré)
    for j in range(n):
        if L[j,j] != 0:
            L[j,j] = np.sqrt(L[j,j])
        for k in range(j):
            if L[j,k] != 0:
                for i in range(j+1, n):
                    if L[i,k] != 0:
                        L[i,j] = L[i,j] - L[i,k] * L[j,k]

        for i in range(j+1,n):
            if L[i,j] != 0:
                L[i,j] = L[i,j] / L[j,j]
                L[i,i] = L[i,i] - L[i,j] * L[i,j]

        for i in range(j+1,n):
            L[j,i] = 0

    # c  = http://sparse.tamu.edu/Nasa/nasa2146

    x = np.ones_like(b)
    y = np.ones_like(b)
    ro = b - a@x
    rn = np.copy(ro)
    M = np.linalg.inv(L)
    zo = np.copy(ro)
    zn = np.copy(ro)
    p = np.copy(zo)
    ap = np.zeros_like(b)
    app = np.zeros_like(b)
    tol = np.finfo(np.float64).resolution

    for _ in range(n):
        y = np.linalg.solve(L, r)
        z = np.linalg.solve(L.T, z)
        ap = a @ p
        app = ap @ p
        rho0 = ro @ zo
        alpha = rho0 / app
        rn = ro - alpha * ap
        rho1 = rn @ z
        beta = rho1 / rho0
        p = z + beta * p
        x = x + alpha * p
        if np.linalg.norm(rn, ord=np.inf) <= tol:
            break
        else:
            rho0 = rho1
            ro = rn

    return x


def incomplete_cholesky(a, tol=1e-8):
    n = a.shape[0]
    L = np.zeros_like(a)
    
    for i in range(n):
        for j in range(i):
            sum_ = a[i, j] - np.dot(L[i, :j], L[j, :j])
            if abs(sum_) < tol:
                L[i, j] = 0
            else:
                L[i, j] = sum_ / L[j, j]
        
        sum_ = a[i, i] - np.dot(L[i, :i], L[i, :i])
        if sum_ > tol:
            L[i, i] = np.sqrt(sum_)
        else:
            L[i, i] = 0
    
    return L

def GradientConjuguePreconditionneIC2(a, b, tol=1e-8, max_iter=1000):
    n = a.shape[0]
    x = np.zeros_like(b)
    r = b - a @ x
    L = incomplete_cholesky(a)
    
    def apply_preconditioner(r):
        y = np.linalg.solve(L, r)
        z = np.linalg.solve(L.T, y)
        return z
    
    z = apply_preconditioner(r)
    p = z.copy()
    rsold = np.dot(r, z)
    
    for i in range(max_iter):
        ap = a @ p
        alpha = rsold / np.dot(p, ap)
        x += alpha * p
        r -= alpha * ap
        
        if np.linalg.norm(r) < tol:
            break
        
        z = apply_preconditioner(r)
        rsnew = np.dot(r, z)
        p = z + (rsnew / rsold) * p
        rsold = rsnew
        
        print(f"Iteration {i+1}: Residual norm = {np.linalg.norm(r)}")
    
    return x

# http://sparse.tamu.edu/aCUSIM/Pres_Poisson
# Parameters:
# source : str or file-like
# Matrix Market filename (extension .mtx) or open file-like object
# Returns:
#     rows : int
#         Number of matrix rows.
#     cols : int
#         Number of matrix columns.
#     entries : int
#         Number of non-zero entries of a sparse matrix or rows*cols for a dense matrix.
#     format : str
#         Either ‘coordinate’ or ‘array’.
#     field : str
#         Either ‘real’, ‘complex’, ‘pattern’, or ‘integer’.
#     symmetry : str
#         Either ‘general’, ‘symmetric’, ‘skew-symmetric’, or ‘hermitian’.

def minres(
    a: np.ndarray,
    b: np.ndarray,
    k: int = 50,
    tol: float = 1e-8
    ) -> np.ndarray:

    n = a.shape[0]
    if (n != b.shape[0]):
        print("Ordre de a != ordre de b.")
        exit()



# http://www.ist.aichi-pu.ac.jp/person/sogabe/thesis.pdf
# an extension of the conjugate residual method to nonsymmetric linear systems
def bicr(
    a: np.ndarray,
    b: np.ndarray,
    k: int,
    tol: float = np.finfo(np.float64).eps
    ) -> np.ndarray:

    err1:str = "a n'est pas en 2D !"
    err2:str = "a est vide !"
    err3:str = "a est rectangulaire !"
    err4:str = "Ordre de a != ordre de b."

    if a.ndim != 2:
        raise np.linalg.LinalgError(err1)

    if a.size == 0:
        raise np.linalg.LinalgError(err2)

    n, m = a.shape
    if n != m:
        raise np.linalg.LinalgError(err3)

    if n != b.shape[0]:
        raise np.linalg.LinalgError(err4)

    x = np.ones(n)
    r0 = b - a@x
    r0star = r0.copy()
    r1 = r0.copy()
    r1star = r0.copy()
    p0 = np.zeros(n)
    p0star = np.zeros(n)
    p1 = np.zeros(n)
    p1star = np.zeros(n)
    alpha = beta = 0.0
    
    for _ in range(k):
        p1 = r1 + beta * p0
        p1star = r1star + beta * p0star
        ap = a@r1 + beta * (a@p0)
        atps = a.T@p1star
        alpha = (np.inner(r1star,a@r1)) / (atps@ap)
        x = x + alpha  *p1
        r1 = r0 - alpha  *ap
        r1star = r0star - alpha*atps
        beta = (np.inner(r1star,a@r1))/(np.inner(r0star,a@r0))
        if np.linalg.norm(r1, ord=np.inf) <= tol*np.linalg.norm(b, ord=np.inf):
            break
        else:
            r0 = r1
            r0star = r1star
            p0 = p1
            p0star = p1star
    return x

def cgs(
    a: np.ndarray,
    b: np.ndarray,
    k: int,
    tol: float = np.finfo(np.float64).eps
    ) -> np.ndarray:

    err1:str = "A n'est pas en 2D !"
    err2:str = "A est vide !"
    err3:str = "A est rectangulaire !"
    err4:str = "Ordre de A != ordre de b."

    if a.ndim != 2:
        raise np.linalg.LinAlgError(err1)

    if a.size == 0:
        raise np.linalg.LinAlgError(err2)

    n, m = a.shape
    if n != m:
        raise np.linalg.LinAlgError(err3)

    if n != b.shape[0]:
        raise np.linalg.LinAlgError(err4)

    x = np.ones(n)
    r = b - a@x
    r0 = r.copy()
    rn = r.copy()
    p = r.copy()
    q = r.copy()
    u = r.copy()
    alpha = beta = rho = 0.0
    
    for _ in range(k):
        rho = np.inner(r0,r)
        ap = a@p
        alpha = rho / np.inner(r,ap)
        q = u - alpha * ap
        upq = u + q
        auq = a@upq
        x = x + alpha * upq
        rn = rn - alpha * auq
        beta = np.inner(rn,r) / rho
        u = rn + beta * q
        p = u + beta * (q + beta * p)
        err = np.linalg.norm(r0, ord=np.inf)
        if np.isclose(err, 0.0, atol=tol) == True:
            break
        else:
            r0 = rn

    return x.astype(np.float64)

def cgs2(
    a: np.ndarray,
    b: np.ndarray,
    k: int = 5,
    tol: float = 1e-8
    ) -> np.ndarray:

    n = a.shape[0]
    if (n != b.shape[0]):
        print("Ordre de a != ordre de b.")
        exit()

    x0 = np.ones(n)
    x1 = np.ones(n)
    r0 = b - a@x0
    rn = r0.copy()
    c = np.zeros(n)
    s = np.zeros(n)
    u0 = np.zeros(n)
    u1 = np.zeros(n)
    v = np.zeros(n)
    w0 = np.zeros(n)
    w1 = np.zeros(n)
    alpha = beta = rho = sigma = 1.0
    alphat = betat = rhot = sigmat = 1.0
    
    for i in range(k):
        print("-------IT-------")
        rho = rn @ r0
        # if alphat == np.inf or alphat == 0.:
        #     alphat = 
        beta = (-1.0 / alphat) *  (rho / sigma)
        v = r0 - beta * u0
        rhot = r0 @ s
        betat = (-1.0 / alpha) * (rhot / sigmat)
        t = r0 - betat * s
        print(f"{rho=}")
        print(f"{rhot=}")
        print(f"{alpha=}")
        print(f"{alphat=}")
        print(f"{beta=}")
        print(f"{betat=}")
        print(f"{t=}")
        print(f"{u0=}")
        print(f"{u1=}")
        w1 = t - beta * (u0 - betat * w0)
        print(f"{w0=}")
        print(f"{w1=}")
        c = a @ w1
        print(f"{c=}")
        sigma = c @ r0
        alpha = rho / sigma
        s = t - alpha * c
        sigmat = c @ s
        alphat = rhot / sigmat
        u1 = v - alphat * c
        x1 = x0 + alpha * v + alphat * s
        rn = r0 - a @ (alpha * v + alphat * s)
        err = np.linalg.norm(x1-x0, ord=np.inf)
        if np.isclose(err, 1e-8, atol=tol) == True:
            break
        else:
            r0 = rn
            x0 = x1
            w0 = w1
            u0 = u1
    return x

def bicgstab(
    a: np.ndarray,
    b: np.ndarray,
    k: int,
    tol: float = np.finfo(np.float64).eps
    ) -> np.ndarray:

    err1:str = "a n'est pas en 2D !"
    err2:str = "a est vide !"
    err3:str = "a est rectangulaire !"
    err4:str = "Ordre de a != ordre de b."

    if a.ndim != 2:
        raise np.linalg.LinalgError(err1)

    if a.size == 0:
        raise np.linalg.LinalgError(err2)

    n, m = a.shape
    if n != m:
        raise np.linalg.LinalgError(err3)

    if n != b.shape[0]:
        raise np.linalg.LinalgError(err4)

    xn = np.ones(n)
    r0 = b - a @ xn
    rn = dn = r0
    r0_norm = np.linalg.norm(r0)
    dn = np.ones(n)
    sn = np.ones(n)

    for _ in range(k):
        adn = a @ dn
        rr0 = np.inner(rn,r0)
        alpha = rr0 / (adn @ r0)
        sn = rn - alpha * adn
        asn = a @ sn
        a2s = asn @ sn
        as2 = asn @ asn
        omega = a2s / as2
        xn = xn + alpha * dn + omega * sn
        rn = sn - omega * asn
        beta = (alpha / omega) * (np.inner(rn,r0) / rr0)
        dn = rn + beta * (dn - omega * adn)
        if np.dot(rn,rn) < tol**2 * r0_norm**2:
            break

    return xn

def bicgstabl(
    a: np.ndarray,
    b: np.ndarray,
    l: int,
    ite: int = 20,
    tol: float = np.finfo(np.float32).resolution
    ) -> np.ndarray:

    assert a is not None
    assert b is not None

    if l == None or l <= 1:
        l = 2

    n = a.shape[0]
    if (n != b.shape[0]):
        print("Ordre de a != ordre de b.")
        exit()

    # k = -l

    # mem = np.empty((2 * l + 5, n))
    x = np.ones_like(b)
    rHat = np.empty(l+1) # résidus
    uHat = np.empty(l+1) # directions
    r0 = b - a @ x
    rHat = np.copy(r0)
    uHat = np.copy(r0)
    res = np.linalg.norm(r0)
    tau = np.zeros((l,l))
    sigma = np.empty(l)
    gamma0 = np.empty(l)
    gamma1 = np.empty(l)
    gamma2 = np.empty(l)
    gamma = 0
    rho0 = -1
    rho1 = -1
    alpha = 0
    beta = 0
    omega = 1
    it:int = 1

    for _ in range(ite):
        rho0 *= -omega
        for j in range(l):
            print("rHat[j] = ", rHat[j])
            print("r0 = ", r0)
            rho1 = np.inner(rHat[j], r0)
            print("rho1 = ", rho1)
            if np.abs(rho0) < tol:
                print("Breakdown No 1")
                exit()
            beta = alpha * (rho1 / rho0)
            rho0 = rho1

            for i in range(j+1):
                uHat[i] = rHat[i] - beta * uHat[i]
                
            uHat[j+1] = a @ uHat[j]
            gamma = np.dot(uHat[j+1], r0)
            if np.isclose(gamma, 0.0, atol=tol) == True:
                print("Breakdown No 2")
                exit()
            alpha = rho0 / gamma

            for i in range(j+1):
                rHat[i] = rHat[i] - alpha * uHat[i+1]
            rHat[j+1] = a @ rHat[j]
            x = x + alpha * uHat[0]

        for j in range(l):
            if np.isclose(sigma[j], 0.0, atol=tol) == True:
                print("Breakdown No 3")
                exit()
            for i in range(j):
                tau[i, j] = (1 / sigma[i]) * (rHat[j+1] @ rHat[i+1])
                rHat[j+1] = rHat[j+1] - tau[i, j] * rHat[i+1]
            sigma[j] = rHat[j+1] @ rHat[j+1]
            gamma1[j] = (1 / sigma[j]) * (rHat[0] @ rHat[j+1])

        gamma0[l-1] = gamma1[l-1]
        omega = gamma0[l-1]
        for j in range(l-2, -1, -1):
            for i in range(j+1, l):
                gamma0[j] = gamma1[j] - np.dot(tau[j,i], gamma0[i])

        for j in range(l-1):
            for i in range(j+1, l-1):
                gamma2[j] = gamma0[j+1] + np.dot(tau[j,i], gamma0[i+1])
                
        x = x + gamma0[0] * rHat[0]
        rHat[0] = rHat[0] - gamma1[l-1] * rHat[l]
        uHat[0] = uHat[0] - gamma0[l-1] * uHat[l]

        for j in range(1,l):
            uHat[0] = uHat[0] - gamma0[j-1] * uHat[j]
            x = x + gamma2[j-1] * rHat[j]
            rHat[0] = rHat[0] - gamma1[j-1] * rHat[j]

        res = b - a @ x
        err = np.linalg.norm(res, ord=np.inf)
        if np.isclose(err, 0.0, atol=tol) == True or it >= ite:
            exit()
        else:
            it += 1

    return x, it


def qmrcgstab(
    a: np.ndarray,
    b: np.ndarray,
    k: int,
    tol: float = np.finfo(np.float64).eps
    ) -> np.ndarray:

    err1:str = "a n'est pas en 2D !"
    err2:str = "a est vide !"
    err3:str = "a est rectangulaire !"
    err4:str = "Ordre de a != ordre de b."

    if a.ndim != 2:
        raise np.linalg.LinalgError(err1)

    if a.size == 0:
        raise np.linalg.LinalgError(err2)

    n, m = a.shape
    if n != m:
        raise np.linalg.LinalgError(err3)

    if n != b.shape[0]:
        raise np.linalg.LinalgError(err4)

    x = np.ones(n)
    xn = np.ones(n)
    r0 = b-a@x
    rn = np.copy(r0)
    dn = np.zeros(n)
    sn = np.copy(r0)
    pn = np.zeros(n)
    nu = np.zeros(n)
    tau = tau_tilde = np.fabs(np.linalg.norm(r0, ord=np.inf))
    alpha = beta = omega = 1.0
    eta = eta_tilde = 0.0
    rho1 = rho2 = 1.0
    theta = theta_tilde = np.linalg.norm(rn, ord=np.inf) / tau_tilde

    for _ in range(k):
        rho2 = np.inner(r0,rn)
        beta = (rho2 * alpha) / (rho1 * omega)
        pn = rn + beta * (pn - omega * nu)
        nu = a@pn
        rho1 = np.inner(r0,nu)
        alpha = rho2 / rho1
        sn = rn - alpha * nu
        theta_tilde = np.linalg.norm(sn, ord=np.inf) / tau
        c = 1.0 / np.sqrt(1.0 + theta_tilde**2)
        tau_tilde = tau * theta_tilde * c
        eta_tilde = c**2 * alpha
        dn = pn + ((theta**2 * eta) / alpha) * dn
        xn = x + eta_tilde * dn
        t = a@sn
        omega = np.inner(sn,t) / np.inner(t,t)
        rn = sn - omega * t
        theta = np.linalg.norm(rn, ord=np.inf) / tau_tilde
        c = 1.0 / np.sqrt(1.0 + theta**2)
        tau = tau_tilde * theta * c
        eta = c**2 * omega
        dn = sn + ((theta_tilde**2 * eta_tilde) / omega) * dn
        x = xn + eta * dn
        err = np.linalg.norm(b-a@x, ord=np.inf)*np.fabs(tau)
        if np.isclose(err, 0.0, atol=tol) == True:
           break
        else:
            rho1 = rho2
    return x

def qmrcgstab2(
    a: np.ndarray,
    b: np.ndarray,
    k: int,
    tol: np.float64
    ) -> np.ndarray:

    err1:str = "A n'est pas en 2D !"
    err2:str = "A est vide !"
    err3:str = "A est rectangulaire !"
    err4:str = "Ordre de A != ordre de b."

    if a.ndim != 2:
        raise np.linalg.LinAlgError(err1)

    if a.size == 0:
        raise np.linalg.LinAlgError(err2)

    n, m = a.shape
    if n != m:
        raise np.linalg.LinAlgError(err3)

    if n != b.shape[0]:
        raise np.linalg.LinAlgError(err4)

    x = xn = np.ones(n)
    r0 = b - a @ x
    rn = sn = np.copy(r0)
    dn = pn = nu = np.zeros(n)
    alpha = beta = omega = 1
    eta = eta_tilde = 0
    rho1 = rho2 = 1
    tau = tau_tilde = np.fabs(np.linalg.norm(r0))
    theta = theta_tilde = np.linalg.norm(rn) / tau_tilde

    for _ in range(k):
        rho2 = np.inner(r0,rn)
        beta = (rho2 * alpha) / (rho1 * omega)
        pn = rn + beta * (pn - omega * nu)
        nu = a @ pn
        rho1 = np.inner(r0,nu)
        alpha = rho2 / rho1
        sn = rn - alpha * nu
        theta_tilde = np.linalg.norm(sn) / tau
        c = np.reciprocal(np.sqrt(1 + theta_tilde**2))
        tau_tilde = tau * theta_tilde * c
        eta_tilde = c**2 * alpha
        dn = pn + ((theta**2 * eta) / alpha) * dn
        xn = x + eta_tilde * dn
        t = a @ sn
        omega = np.inner(sn,sn) / np.inner(sn,t)
        rn = sn - omega * t
        theta = np.linalg.norm(rn) / tau_tilde
        c = np.reciprocal(np.sqrt(1 + theta**2))
        tau = tau_tilde * theta * c
        eta = c**2 * omega
        dn = sn + ((theta_tilde**2 * eta_tilde) / omega) * dn
        x = xn + eta * dn
        err = np.linalg.norm(b - a @ x) * np.fabs(tau)

        if np.isclose(err, 0.0, atol=tol) == True:
           break
        else:
            rho1 = rho2
            
    return x.astype(np.float64)

#orthomin risque de breakdown, mais consomme peu
#orthodir pas de risque de breakdown, mais consomme plus

def orthomin(
    a: np.ndarray,
    b: np.ndarray,
    k: int = 50,
    tol: float = 1e-8
    ) -> np.ndarray:

    n = a.shape[0]
    if (n != b.shape[0]):
        print("Ordre de a != ordre de b.")
        exit()
    xn = np.ones(n)
    rn0 = rn1 = rn2 = np.array(b-a@xn)
    rt0 = rt1 = rt2 = np.copy(rn0)
    pn0 = pt0 = np.copy(rn0)
    pn1 = pt1 = np.copy(rn0)
    alpha = beta = err = np.float128(0.0)

    for i in range(k):
        ap = a@pn1
        alpha = (np.dot(rn1,rt1)) / (np.dot(ap,pt1))
        pn1 = rn1 + alpha*pn0
        pt1 = rt1 + alpha*pt0
        beta = (np.dot(rn1,rt1)) / (np.dot(rn0,rt0))
        rn2 = rn1 - alpha * a*pn1
        rt2 = rt1 - alpha * a.T*pt1
        xn = xn + alpha * pn1
        err = np.linalg.norm(b-a@xn, ord=np.inf)
        if np.isclose(err, 0.0, atol=tol) == True:
            break
        else:
            rn0 = rn1
            rn1 = rn2
            rt0 = rt1
            rt1 = rt2
            pn0 = pn1
            pt0 = pt1
    return xn


def gpbicg(
    a: np.ndarray,
    b: np.ndarray,
    k: int,
    tol: float = np.finfo(np.float64).eps
    ) -> np.ndarray:

    err1:str = "a n'est pas en 2D !"
    err2:str = "a est vide !"
    err3:str = "a est rectangulaire !"
    err4:str = "Ordre de a != ordre de b."

    if a.ndim != 2:
        raise np.linalg.LinalgError(err1)

    if a.size == 0:
        raise np.linalg.LinalgError(err2)

    n, m = a.shape
    if n != m:
        raise np.linalg.LinalgError(err3)

    if n != b.shape[0]:
        raise np.linalg.LinalgError(err4)

    xn = np.ones(n, dtype = np.float64)
    ro = b - a @ xn
    r0star = np.copy(ro)
    rn = np.ones(n, dtype = np.float64)
    pn = to = tn = un = wn = yn = zn = np.copy(ro)
    alpha = beta = eta = zeta = np.float64(0.0)
    d1 = d2 = e1 = e2 = b1 = b2 = np.float64(0.0)

    for i in range(k):
        pn = ro + beta * (pn - un)
        apn = a@pn
        alpha = np.inner(r0star,rn) / np.inner(r0star,apn)
        yn = to - rn - alpha * wn + alpha * apn
        tn = ro - alpha * apn
        atn = a @ tn
        if i == 0:
            zeta = (atn@tn) / (atn@atn)
            eta = 0
        else:
            d1 = (atn@atn) * np.inner(yn,yn)
            d2 = np.inner(yn,atn) * np.inner(atn,yn)
            denom = d1 - d2
            z1 = np.inner(yn,yn) * np.inner(atn,tn)
            z2 = np.inner(yn,tn) * np.inner(atn,yn)
            zeta = z1 - z2
            zeta /= denom
            e1 = (atn@atn) * np.inner(yn,tn)
            e2 = np.inner(yn,atn) * np.inner(atn,tn)
            eta = e1 - e2
            eta /= denom
        un = zeta * apn + eta * (to - ro + beta * un)
        zn = zeta * ro + eta * zn - alpha * un
        xn += alpha * pn + zn
        rn = tn - eta * yn - zeta * atn
        b1 = (alpha / zeta)
        b2 = ((np.inner(r0star,rn) / np.inner(r0star,ro)))
        beta = b1 * b2
        wn = atn + beta * apn
        norme1 = np.linalg.norm(a@xn)
        norme2 = tol * np.linalg.norm(b)
        if norme1 <= norme2:
            break
        else:
            ro = rn
            to = tn

    return xn



def tfqmr(
    a: np.ndarray,
    b: np.ndarray,
    k: int = 50,
    tol: float = 1e-8
    ) -> np.ndarray:

    n = a.shape[0]
    if (n != b.shape[0]):
        print("Ordre de a != ordre de b.")
        exit()

    x = np.ones(n)
    r0 = b - a@x
    w = np.copy(r0)
    y1 = np.copy(r0)
    y2 = y1
    v = a@y1
    d = np.zeros(n)
    tau = np.linalg.norm(r0, ord=np.inf)
    alpha = beta = theta = eta = 0.0
    rho1: np.float128 = np.inner(r0, r0)
    rho2: np.float128 = np.inner(r0, r0)

    for m in range(k):
        if np.isclose(rho1, 0.0, atol=tol) == True or np.isclose(rho2, 0.0, atol=tol) == True:
            print("TFQMR breakdown.")
            exit()
        sigma = np.inner(r0, v)
        alpha = rho1 / sigma
        y2 = y1 - alpha * v
        for m in range(2*i-1,2*i):
            w -= alpha * (a@y1)
            theta = np.linalg.norm(w, ord=np.inf) / tau
            c = 1.0 / np.sqrt(1.0 + theta**2) # c = 1.0 / np.hypot(1.0, theta)
            tau *= theta * c
            eta = c**2 * alpha
            d *= ((theta**2 * eta) / alpha)
            d += y1
            x += eta * d
            err = tau * np.sqrt(m+1)
            if np.isclose(err, 0.0, atol=tol) == True:
                break
        r0 = b - a@x
        rho2 = np.inner(r0, w)
        beta = rho2 / rho1
        y1 = w + beta * y2
        v = a@y2 + beta * (a@y1 + beta * v)
        y1 = y2
        rho1 = rho2

    return x


def crs(
    a: np.ndarray,
    b: np.ndarray,
    k: int,
    tol: float = np.finfo(np.float64).eps
    ) -> np.ndarray:

    err1:str = "A n'est pas en 2D !"
    err2:str = "A est vide !"
    err3:str = "A est rectangulaire !"
    err4:str = "Ordre de A != ordre de b."

    if a.ndim != 2:
        raise np.linalg.LinAlgError(err1)

    if a.size == 0:
        raise np.linalg.LinAlgError(err2)

    n, m = a.shape
    if n != m:
        raise np.linalg.LinAlgError(err3)

    if n != b.shape[0]:
        raise np.linalg.LinAlgError(err4)

    x0 = np.ones(n)
    xn = np.ones(n)
    r0 = b-a@x0
    r0star = np.copy(r0)
    rn = np.copy(r0)
    qn = np.copy(r0)
    p0 = np.copy(r0)
    pn = np.copy(r0)
    u0 = np.copy(r0)
    un = np.copy(r0)
    alpha = beta = 0.0

    for _ in range(k):
        atrs = a.T@r0star
        alpha = np.inner(r0,atrs) / np.inner(a@pn,atrs)
        qn = u0 - alpha * a@pn
        xn = x0 + alpha * (u0 + qn)
        rn = r0 - alpha * a@(u0 + qn)
        beta = np.inner(rn,atrs) / np.inner(r0,atrs)
        un = rn + beta * qn
        pn = un + beta * (qn + beta*p0)
        err = np.linalg.norm(rn)
        if np.isclose(err, 0.0, atol=tol) == True:
            break
        else:
            r0 = rn
            p0 = pn
            u0 = un
            x0 = xn

    return xn.astype(np.float64)

def cors(
    a: np.ndarray,
    b: np.ndarray,
    k: int,
    tol: float = np.finfo(np.float64).eps
    ) -> np.ndarray:

    err1:str = "a n'est pas en 2D !"
    err2:str = "a est vide !"
    err3:str = "a est rectangulaire !"
    err4:str = "Ordre de a != ordre de b."

    if a.ndim != 2:
        raise np.linalg.LinalgError(err1)

    if a.size == 0:
        raise np.linalg.LinalgError(err2)

    n, m = a.shape
    if n != m:
        raise np.linalg.LinalgError(err3)

    if n != b.shape[0]:
        raise np.linalg.LinalgError(err4)

    x0 = x1 = np.ones(n)
    r0 = r1 = b-a@x0
    r0star = np.inner(a,r0)
    rhat = np.inner(a,r0)
    e0 = e1 = np.copy(r0)
    f = h = np.zeros(n)
    qh = np.copy(r0)
    d0 = d1 = np.copy(rhat)
    q0 = q1 = np.copy(rhat)
    rho0 = np.inner(r0star, rhat)
    alpha = beta = 0.0

    for _ in range(k):
        qh = a@q0
        rho0 = np.inner(r0star, rhat)
        alpha = rho0 / np.inner(r0star,qh)
        h = e0 - alpha * q0
        f = d0 - alpha * qh
        x1 = x0 + alpha * (e0 + h)
        r1 = r0 - alpha * (d0 + f)
        rhat = a@r1
        rho1 = np.inner(r0star,rhat)
        beta = rho1 / rho0
        e1 = r1 + beta * h
        d1 = rhat + beta * f
        q1 = d1 + beta * (f + beta * q0)
        rn = np.linalg.norm(b - a @ x1)
        if np.isclose(rn, 0.0, atol=tol) == True:
            break
        else:
            rho1 = rho0
            d0 = d1
            e0 = e1
            q0 = q1
            x0 = x1
            r0 = r1

    return x1

def bicorstab2(
    a: np.ndarray,
    b: np.ndarray,
    k: int = 50,
    tol: float = 1e-8
    ) -> np.ndarray:

    n = a.shape[0]
    if (n != b.shape[0]):
        print("Ordre de a != ordre de b.")
        exit()

def tfqmors(
    a: np.ndarray,
    b: np.ndarray,
    k: int = 50,
    tol: float = 1e-8
    ) -> np.ndarray:

    n = a.shape[0]
    if (n != b.shape[0]):
        print("Ordre de a != ordre de b.")
        exit()

    x0 = x1 = np.ones(n)
    r0 = b-a@x0
    r0star = np.inner(a,r0)
    r0h = np.dot(a,r0)
    rho0 = rho1 = np.inner(r0star,r0h)
    wj = np.copy(r0)
    wjh = np.copy(r0)
    wk0 = wk1 = np.copy(r0)
    wk0h = wk1h = np.copy(r0)
    qn = np.inner(a,r0star)
    qnh = np.copy(r0h)
    y0 = y1 = y2 = np.copy(r0)
    y0h = y1h = y2h = np.copy(r0)
    theta = eta = 0.0
    tau = np.linalg.norm(r0, ord=np.inf)
    d = np.zeros(n)

    for j in range(k):
        alpha = rho0 / np.inner(r0star,qnh)
        y1 = y0 - alpha * qn
        y1h = y0h - alpha * qnh
        for k in range(2*j-1,2*j):
            wk1 = wk0 - alpha * y1h
            if k == 2*j:
                y1h = a@y1h
            wk1h = wk0h - alpha * y1h
            d = y1 + d*((theta**2 * eta) / alpha)
            theta = np.linalg.norm(wk1, ord=np.inf) / tau
            c = np.float128(1.0 / np.sqrt(1.0+theta**2))
            tau *= alpha * c
            eta = c**2 * alpha
            x1 = x0 + eta*d
            err = np.linalg.norm(b-a@x1, ord=np.inf)
            if np.isclose(err, 0.0, atol=tol) == True:
                break
            else:
                rho0 = rho1
                wk0 = wk1
                wk0h = wk1h
                x0 = x1
                y1 = y2
                y0 = y1
                y1h = y2h
                y0h = y1h
        rho1 = np.inner(r0star,wj)
        beta = rho1 / rho0
        y2 = wj + beta * y1
        y2h = wjh + beta * y1h
        qn = y2h + beta * (y1h + beta * qn)
        y2h = a@y2h
        qnh = y2h + beta * (y1h + beta * qnh)
    return x

# def rotation(x,y,c,s):
#     if y == 0:
#         c = 1.0
#         s = 0.0
#     elif abs(y) > abs(x):
#         z = x/y
#         s = 1.0 / (1+z**2)
#         c = z * s
#     else:
#         z = y/x
#         s = 1.0 / (1+z**2)
#         s = z * c

# def gmres(a,b,n,l,err):
#     n = len(a)
#     m = len(b)

#     if (n != m):
#         print('error dim(a) != b\n')

# def arnoldi(a,v):
#     n = len(a)
#     v = np.zeros((n,n),dtype='float64')
#     w = np.zeros((n,n),dtype='float64')
#     v = (1/np.linalg.norm(v, ord=np.inf)*v
    # for i in range(0,n-1):
    #     w[i] = np.dot(a[:,i], v[:,j])
    #     for j in range(0,i):
    #         a[i,j] = np.dot(v,w)
    #         w[j] -= a[i,j]*v[j]
    #     h[j,j] = np.linalg.norm(v[:,j], ord=np.inf)
    #     v[i:,j] = v[:,j] / r[j,j]

    #return q, r

def qr_gram_schmidt_classique(
        a: np.ndarray
        ) -> tuple[np.ndarray,np.ndarray]:
    n = a.shape[1]
    q = np.zeros((n,n))
    r = np.zeros((n,n))

    for j in range(n):
        q[:,j] = a[:,j]
        for i in range(j):
            r[i,j] = a[:,j] @ q[:,i]
            q[:,j] -= r[i,j] * q[:,i]
        r[j,j] = np.linalg.norm(q[:,j])
        q[:,j] /= r[j,j]
        if np.array_equal(q, np.zeros(n)):
            raise np.linalg.LinalgError("|Q|=0.")

    return q, r

def qr_gram_schmidt_modifiee(
        a: np.ndarray
        ) -> tuple[np.ndarray,np.ndarray]:
    n = a.shape[1]
    q = np.zeros((n,n))
    r = np.zeros((n,n))

    for i in range(n):
        q[:,i] = a[:,i]
    for i in range(n):
        r[i,i] = np.linalg.norm(q[:,i])
        q[:,i] /= r[i,i]
        for j in range(i+1,n):
            r[i,j] = q[:,i] @ q[:,j]
            q[:,j] -= r[i,j] * q[:,i]
            if np.array_equal(q, np.zeros(n)):
                raise np.linalg.LinalgError("|Q|=0.")

    return q,r

def quotient_rayleigh(a):
    n = len(a)
# [n,∼]=size(a);
# x0=zeros(n,1);
# % initialize initial vector x0 which has norm 1
# x0(n)=1;
# tol = 1e-10;
# xi = x0/norm(x0,2);
# i=0;
# % initialize Rayleigh Quotient for x0
# rq = (xi'*a*xi)/(xi'*xi);
# while norm((a*xi-rq*xi),2) > tol
# yi = (a-rq*eye(size(a)))\xi;
# xi=yi/norm(yi,2);
# rq = (xi'*a*xi)/(xi'*xi)
# i=i+1;
# end
    return 0

def hessenberg(a):
    n = len(a)
    x = np.ones(n)
    u = np.ones(n)

# for k=1:n-2
# x=a(k+1:n,k)
# u=x
# u(1) = u(1)+sign(x(1))*norm(x)
# u=u/norm (u)
# P=eye(n-k) - 2*(u*u')
# a(k+1:n,k:n) = P*a(k +1:n,k:n)
# a(1:n,k+1:n) = a(1:n,k+1:n)*P

    return 0

# function HessenbergReduction( a::Matrix )
#     # Reduce a to a Hessenberg matrix H so that a and H are similar:
    
#     n = size(a, 1)
#     H = a
#     if ( n > 2 )
#         a1 = a[2:n, 1]
#         e1 = zeros(n-1); e1[1] = 1
#         sgn = sign(a1[1])
#         v = (a1 + sgn*norm(a1)*e1); v = v./norm(v)
#         Q1 = eye(n-1) - 2*(v*v')
#         a[2:n,1] = Q1*a[2:n,1]
#         a[1,2:n] = Q1*a[1,2:n]
#         a[2:n,2:n] = Q1*a[2:n,2:n]*Q1' 
#         H = HessenbergReduction( a[2:n,2:n] )
#     else
#         H = copy(a)
#     end
#    return a
# end

# R = H(n)H(n-1)H(...)H(2)H(1)a
# Q = H(1)H(2)H(...)H(n-1)H(n)
        # if d < np.finfo(float).eps:
        #     quit()
def qr_householder(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = m.shape[0]
    Q = np.identity(n)
    R = np.copy(m)
    for i in range(n-1):
        x = R[i:,i]
        e = np.zeros_like(x)
        e[0] = 1
        v = x + np.copysign(np.linalg.norm(x, ord=np.inf), x[0]) * e
        H = np.identity(n)
        H[i:,i:] -= ((2.0 * np.outer(v,v.T)) / np.dot(v.T,v))
        R = H@R
        Q = Q@H

    return Q,R

# https://github.com/GeeKboy2/Image-Compression-through-SVD-Factorization/blob/main/src/part1.py
def householder2(U, V):
	if np.array_equal(U, V):
		return None
	diff = U - V
	res = diff / np.linalg.norm(diff)
	N = res.reshape(len(U), 1)
	identity = np.eye(len(U))
	if N is None:
		return identity, None
	return identity - 2 * (N @ N.T), N


def householder_reflecteur(x: np.ndarray) -> np.ndarray:
    n:int = len(x)
    u = np.zeros(n)
    u = np.copy(x)
    nx = np.linalg.norm(x, ord=np.inf)
    u[0] = nx + np.fabs(x[0])
    beta = u[0] / nx
    if x[0] < 0:
        u[0] = -u[0]
    else:
        nx = -nx
    u = u/u[0]

    return u

def qr_givens(a):
    n = a.shape[0]
    Q = np.identity(n)
    R = a.copy()

    for j in range(n):
        for i in range(n-1, j, -1):
            rij = R[i,j]
            rjj = R[j,j]
            if rij != 0:
                rho = np.sqrt(rjj**2 + rij**2)
                c = rjj / rho
                s = -rij / rho

                G = np.identity(n)
                G[j,j] = c
                G[i,i] = c
                G[i,j] = s
                G[j,i] = -s

                R = G @ R
                Q = Q @ G.T

    return Q, R

def givens_cos(m,i,j):
    if m[i,i] == 0 and m[i,j] == 0:
        return 1
    else:
        return m[i,i]/np.sqrt(m[i,i]*m[i,i]+m[j,i]*m[j,i])

def givens_sin(m,i,j):
    if m[i,i] == 0 and m[i,j] == 0:
        return 0
    else:
        return m[j,i]/np.sqrt(m[i,i]*m[i,i]+m[j,i]*m[j,i])

def psi(a):
    n=len(a)
    p=0
    for i in range(1,n):
        for j in range(i):
            p+=a[i][j]**2
    return p

def jacobi(a: np.ndarray, k: int = 100) -> np.ndarray:
    n:int = a.shape[0]
    theta: np.float128 = 0
    p: int = 0
    q: int = 0
    it = 0
    tol: np.float32 = np.finfo(np.float32).resolution

    while it < k:
        val_max_hors_diag = 0
        p = 0
        q = 0
        
        for i in range(n):
            for j in range(i+1, n):
                if np.abs(a[i, j]) > val_max_hors_diag:
                    val_max_hors_diag = np.abs(a[i, j])
                    p = i
                    q = j
        
        if val_max_hors_diag < tol:
            break
        
        a_pp = a[p, p]
        a_pq = a[p, q]
        a_qq = a[q, q]
        
        if val_max_hors_diag == 0:
            theta = 0
        elif a_pp == a_qq:
            if a_pp > tol:
                theta = np.pi / 4
            else:
                theta = -np.pi / 4
        else:
            theta = 0.5 * np.arctan(2 * a_pq / (a_pp - a_qq))
        
        
        apq = a[p, q]
        app = a[p, p]
        aqq = a[q, q]

        # c = np.cos(theta)
        # s = np.sin(theta)

        c = (app - aqq) / (2 * apq)
        t = np.reciprocal((np.fabs(c) + np.sqrt(1.0 + c**2))) # 1 / ...
        if (c < 0.0):
            t = -t
        c = np.reciprocal(np.sqrt(1.0 + t**2)) # 1 / ...
        s = c * t
        cc = c * c
        ss = s * s
        cs = c * s

        a[p, p] = app * cc - 2 * apq * cs + aqq * ss
        a[q, q] = app * ss + 2 * apq * cs + aqq * cc
        a[p, q] = 0
        a[q, p] = 0
        
        for r in range(n):
            if r != p and r != q:
                arp = a[r, p]
                arq = a[r, q]
                print(arp)
                print(arq)
                a[r, p] = arp * c - arq * s
                a[p, r] = a[r, p]
                a[r, q] = arp * s + arq * c
                a[q, r] = a[r, q]
        
    valeurspropres = np.diag(a)
    
    return valeurspropres

# def jacobi2(a,epsilon,iterations):
#     it = 0
#     n = len(a)
#     mvp = np.identity(n)
#     vvp = np.zeros(n)
#     if (r<0.0):
#         t = -t
#     while (it < iterations):
#         for i in range(0,n-1):
#             for j in range(i+1,n):
#                 #if abs(a[i,j])
#                 d = (a[i,i])/(a[j,j])
#                 if (d < tol):
#                     theta = pi/4.0
#                 else:
#                     theta = 0.5 * atan((2.0*a[i,j])/d)
#                 if theta >= 0.0:
#                     t = 1 / np.sqrt(theta + np.sqrt(theta**2+1.0))
#                 if tau < 0.0:
#                    t = -1 / np.sqrt(-theta + np.sqrt(theta**2+1.0))
#         i += 1
#         r = (a[q,q]-a[p,p]) / (2.0*a[p,q])
#         t = 1.0/(fabs(r)+np.sqrt(1+r**2))
#         c = 1.0/np.sqrt(1+t**2.0)
#         s = t*c
#         rho
#         for i in range(1,r-1):
#         for i in range(r+1,s-1):
#         for i in range(s+1,n):
#         for i in range(1,n):

#     return 0


def power(a: np.array, tol: np.longdouble, iterations: int):
    x = np.ones(len(a), dtype=np.float64)
    y = np.ones(len(a), dtype=np.float64)
    lambda_o = lambda_n = 0
    
    for _ in range(iterations):
        y = a@x
        x = y / np.linalg.norm(y, ord=np.inf)
        lambda_n = np.dot(x.T,y)
        if np.fabs((lambda_n-lambda_o)/lambda_n) <= tol:
            break
        else:
            lambda_o = lambda_n

    return lambda_n, x

def power2(a: np.array, tol: np.longdouble, iterations: int):
    x = np.ones(len(a), dtype=a.dtype)
    for i in range(n):
        x = a @ x
        lambda_1, x = np.norm

# def invpower2(a: np.array, tol: np.longdouble, iterations: int):
#     for i in range(n):
#         x = np.linalg.inv(a) @ x
#         lambda_1, x = np.norm

def invpower(a,tol,iterations):
    x = np.ones(len(a), dtype=np.float64)
    f = np.ones(len(a), dtype=np.float64)
    b = np.ones(len(a), dtype=np.float64)
    lambda_o = lambda_n = 0
    
    L, U = lu_factorisation(a)
    for _ in range(iterations):
        f = Substitutionavant(L, x)
        b = Substitutionarriere(U, f)
        b = b / np.linalg.norm(b, ord=np.inf)
        lambda_n = np.dot(b.T, a@b)
        if np.fabs((lambda_n-lambda_o)/lambda_n) <= tol:
            break
        else:
            lambda_o = lambda_n

    return lambda_n, x

def roots(*coeffs):
    matrix = np.eye(len(coeffs)-1, k=1)
    matrix[:,0] = np.array(coeffs[1:]) / -coeffs[0]
    return np.linalg.eigvals(matrix)

#z = roots(1,1,1,1,1,1)
#print(z)

def reglin(
    x: np.ndarray,
    y: np.ndarray
    ) -> tuple[float, float]:
    n = x.shape[0]
    a = np.zeros((2,2))
    b = np.zeros(n)
    
    for i in range(n):
        xi = x[i]
        yi = y[i]
        a[0][0] += 1
        a[1][0] += xi
        a[0][1] += xi
        a[1][1] += xi**2
        b[0] += y[i]
        b[1] += xi*yi
    
    det = a[0][0]*a[1][1] - a[1][0]*a[0][1]
    x1 = (a[1][1]*b[0] - a[0][1]*b[1]) / det
    x2 = (a[0][0]*b[1] - a[1][0]*b[0]) / det
    
    return x1,x2

def regqua(x: np.ndarray, y: np.ndarray) -> tuple:
    n = x.shape[0]
    a = np.zeros((3,3))
    b = np.zeros(3)
    result = np.zeros(3)

    for i in range(n):
        xi = x[i]
        yi = y[i]
        xi2 = xi**2
        xi3 = xi**3
        a[0][0] += 1
        a[1][0] += xi
        a[2][0] += xi2

        a[0][1] += xi
        a[1][1] += xi2
        a[2][1] += xi3

        a[0][2] += xi2
        a[1][2] += xi3
        a[2][2] += xi2**2

        b[0] += yi
        b[1] += xi*yi
        b[2] += xi2*yi

    print(a)
    resultat = np.linalg.solve(a,b)

    a = result[2]
    b = result[1]
    c = result[0]

    return a, b, c

def interbilin(u,v):
    r = float(0.0)
    for i in range(0,4):
        if (i==0):
            weight = (1-u)*(1-v)
        if (i==1):
            weight = u*(1-v)
        if (i==2):
            weight = u*v
        if (i==3):
            weight = (1-u)*v
        r += weight
    return r

def Lagrange1D(
    x: np.ndarray,
    y: np.ndarray,
    t: float
    ) -> float:

    p = float(0.0)
    li = float(0.0)
    n = np.size(x)

    if n != np.size(y):
        exit()
    for i in range(n):
        li = 1
        for j in range(n):
            if (i != j) and (x[i]-x[j]) != 0:
                try:
                    li *= (t-x[j]) / (x[i]-x[j])
                except ZeroDivisionError:
                    print("Dénominateur nul !")
                    exit()
        p += li * y[i]
    return p

def Chebyshev(a, b, N):
    half = 0.5
    nodes = [0.5*(a+b) + 0.5*(b-a)*np.cos(float(2*i+1)/(2*(n+1))*np.pi) for i in range(N+1)]
    return nodes

#https://people.sc.fsu.edu/~jburkardt/c_src/lagrange_interp_2d/lagrange_interp_2d.c

# def cubicspline(n, points, data):
#     a = np.zeros(n, dtype=np.float64)
#     d = np.zeros(n, dtype=np.float64)
#     c = np.zeros(n, dtype=np.float64)
#     m = np.zeros([n,n], dtype=np.float64)

#     for i in range(n-1):
#         m[i][i] = a(i)
#         m[i][i+1] = 2
#         m[i][i+1] = c(i)

#     return 0

def trapezes(
    f: callable,
    a: np.longdouble,
    b: np.longdouble,
    n: int,
    p: int) -> np.longdouble:
    
    h = np.fabs((b - a) / n)
    t = (f(a) + f(b))
    for i in range(1,n):
        t += 2.0 * f(a + i * h)
    t *= (h * 0.5)

    return np.format_float_positional(np.longdouble(t \
    - np.fabs((b - a)**3)/(12.0 * (n**2))), unique=False, \
    precision=p)


def trapezes_v2(
    f: callable,
    a: np.longdouble,
    b: np.longdouble,
    n: int,
    p: int) -> np.longdouble:

    h = np.fabs((b - a) / n)
    fa = f(a)
    fb = f(b)
    t = h * 0.5 * (fa + 2 * np.sum(np.fromiter((f(a + i * h) \
    for i in range(1,n)), dtype=np.longdouble)) + fb)

    return np.format_float_positional(np.longdouble(t \
    - np.fabs((b - a)**3)/(12.0 * (n**2))), unique=False, \
    precision=p)

    x = np.linspace(0, np.pi, num=n)
    fx = x**3 + 10*x**2 - 4*np.cos(x) + 10
    f = lambda x: x**3 + 10*x**2 - 4*np.cos(x) + 10
    print(f"np.trapezoid(x**5-x**3*np.cos(x**2), 0, pi, {n}) =", np.trapezoid(fx,x))

def simpson13(
	f: callable,
	a: np.longdouble,
	b: np.longdouble,
	n: int
	) -> np.longdouble:
    h = np.fabs((b - a) / n)
    fa:np.longdouble = f(a)
    fb:np.longdouble = f(b)
    s: np.longdouble = fa + fb
    dec = np.finfo(np.longdouble).precision
    for i in range(1,n):
        x = a + i * h
        fx = f(x)
        if i % 2 == 0:
            s +=  4 * fx
        else:
            s += 2 * fx
    return np.format_float_positional(np.longdouble(s * (h / 3.0)), unique=False, precision=dec)

def simpson13_v2(f, a, b, n) -> np.longdouble:
    x = np.linspace(a, b, n)
    h = np.fabs((b - a) / n)
    fx = f(x)
    pas = h / 3.0
    pair = 2 * np.sum(fx[0:n-2:2])
    impair = 4 * np.sum(fx[1:n-1:2])
    s = pas * (pair + impair)
    
    return s

# def simpson13_v3(f: callable, a: np.longdouble, b: np.longdouble, n: int) -> np.longdouble:
# 	x = np.linspace(a, b, n, dtype = np.longdouble)
# 	h = np.fabs((b - a) / n)
#     pas = h / 3
#     fx = f(x)
#     s = pas * (fx[0] + 4 * np.sum(fx[0:n-2:2]) + 2 * np.sum(fx[1:n-1:2]) + fx[n-1])
#     return s

def simpson38(
    f: Callable,
    b: np.longdouble,
    n: int
    ) -> np.longdouble:

    h = np.fabs((b-a)/n)
    s = f(a) + f(b)
    dec = np.finfo(np.longdouble).precision

    for i in range(1,n):
        x = a+i*h
        fx = f(x)
        if i % 3 == 0:
            s += 2 * fx
        else:
            s += 3 * fx
    return np.format_float_positional(np.longdouble(s * 3.0 * (h / 8.0)), unique=False, precision=dec)

def simpson38_v2(f: callable, a: np.float64, b: np.float64, n: int) -> np.float64:
    if n % 3 != 0:
        n += 3 - (n % 3)
    x = np.linspace(a, b, n, dtype = np.float64)
    h = np.fabs((b - a) / n)
    fx = f(x)
    i1 = 3 * np.sum(fx[1:n-1:3]) #[1:n-2:3]
    i2 = 3 * np.sum(fx[2:n-1:3]) #[2:n-1:3]
    i3 = 2 * np.sum(fx[3:n-1:3]) #[3:n-3:3]
    s = (3 / 8 * h) * (f(a) + i1 + i2 + i3 + f(b))
    
    return s

def simpson38_v3(f, a, b, n):
  h = (b - a)/3/n
  return 3*h/8*sum([(f(a + (3*i)*h) + 3*f(a + (3*i + 1)*h) + 3*f(a + (3*i + 2)*h) + f(a + (3*i + 3)*h)) for i in range(int(n))])
##  return (3*h/8) * np.sum(np.fromiter(( h * (f(a+(3*i)) + 3*f(a+(3*i+1)) + 3*f(a+(3*i+2)) + f(a+(3*i+3)))) for i in range(int(n))), dtype=np.float64)

def romberg(
    f: callable,
    a: np.longdouble,
    b: np.longdouble,
    n: int,
    prec: int,
    tol: np.longdouble = sys.float_info.epsilon
    ) -> np.longdouble:

    r = np.zeros((n,n), dtype = np.longdouble)
    r[0,0] = trapezes(f, a, b, 1, prec)
    k = 0
    dec = np.finfo(np.longdouble).precision
    for i in range(1, n+1):
        r[i,0] =  trapezes(f, a, b, 2**i, prec)
        for j in range(1, i-1):
            k = j
            p = 4**j
            d = 1.0 / (p-1.0)
            r[i,j] = (p * r[i,j-1] - r[i-1,j-1]) * d
        if (np.fabs(r[i,k]) - r[i-1,k-1] <= tol):
            break

    return np.format_float_positional(np.longdouble(r[i,k]), unique=False, precision=prec)


def legendre_eval(n,x):
    if n < 0:
        print("legendre_eval : n < 0")
        exit()
    if n == 0:
        p = 1.0
    elif n == 1:
        p = x
    if n > 1:
        return ((2*n-1)*x*legendre_eval(n-1,x)-(n-1) \
                *legendre_eval(n-2,x))/n

    return p

# if (n == 0):
# f = 1e0; d = 0e0
# else:
# f = x; fm1 = 1e0
# for i in range(2,n+1):
# fm2 = fm1; fm1 = f
# f = ((2*i-1)*x*fm1 - (i-1)*fm2)/i
# d = n*(x*f-fm1)/(x*x-1e0) if (x*x-1e0) else 0.5*n*(n+1)*f/x
# return (f, d)

def eval_legendre_d(p,n,x):
    return 0

# faire boucle avec le nombre de racines à calculer
# et approximer la dérivée avec def derivative3(p, x):
def legendre_racines(p,d,n,x):
    x = np.array(float,n)
    x0 = cos((i-0.25)/(n+0.5))
    x = NewtonRaphson(p(x), d(x), x0)
    return 0

def legendre_poids(
    p,
    n: int,
    x: np.float128,
    ) -> np.float128:
    w = np.empty(n)
    x = np.empty(n)

    return 2.0/(n*legendre_eval(n,x)*eval_legendre_d(p,n,x))

import numpy.polynomial.legendre as gl
def gauss_legendre(
    f: callable,
    a: np.longdouble,
    b: np.longdouble
    ) -> np.longdouble:
    
##    w = np.empty(20)
##    x = np.empty(20)
##    dec = np.finfo(np.longdouble).precision
##    
##    x[0] = -0.07652652113349733375; w[0]=0.15275338713072585069
##    x[1] = 0.07652652113349733375; w[1]=0.15275338713072585069
##    x[2] = -0.22778585114164507808; w[2]=0.14917298647260374678
##    x[3] = 0.22778585114164507808; w[3]=0.14917298647260374678
##    x[4] = -0.37370608871541956067; w[4]=0.14209610931838205132
##    x[5] = 0.37370608871541956067; w[5]=0.14209610931838205132
##    x[6] = -0.51086700195082709800; w[6]=0.13168863844917662689
##    x[7] = 0.51086700195082709800; w[7]=0.13168863844917662689
##    x[8] = -0.63605368072651502545; w[8]=0.11819453196151841731
##    x[9] = 0.63605368072651502545; w[9]=0.11819453196151841731
##    x[10] = -0.74633190646015079261; w[10]=0.10193011981724043503
##    x[11] = 0.74633190646015079261; w[11]=0.10193011981724043503
##    x[12] = -0.83911697182221882339; w[12]=0.08327674157670474872
##    x[13] = 0.83911697182221882339; w[13]=0.08327674157670474872
##    x[14] = -0.91223442825132590586; w[14]=0.06267204833410906357
##    x[15] = 0.91223442825132590586; w[15]=0.06267204833410906357
##    x[16] = -0.96397192727791379126; w[16]=0.04060142980038694133
##    x[17] = 0.96397192727791379126; w[17]=0.04060142980038694133
##    x[18] = -0.99312859918509492478; w[18]=0.01761400713915211831
##    x[19] = 0.99312859918509492478; w[19]=0.01761400713915211831

    x, w = gl.leggauss(20)
    s: np.longdouble = 0.0
    h = (b - a) * 0.5
    c = (b + a) * 0.5
#    for i in range(np.size(x)):
#	s += w[i] * f(h * x[i] + c)

    s = np.sum(np.fromiter((w[i] * f(h * x[i] + c) \
        for i in range(np.size(x))), dtype=np.longdouble))


    return np.format_float_positional( \
    np.longdouble(s*h), unique=False, precision=dec)

def mc (f, samples):
    return 0

def mcmetropolis (f, samples):
    return 0

def k(l, m):
    r = ((2.0*l+1.0) * factorial(l-fabs(m)) / (4.0*np.pi*factorial(l+fabs(m))))
    return np.sqrt(r)

def la(l, m, n):
    return 0

def sphericalharmonic(l, m, theta, phi):
    return 0

def Legendre(n,x):
    if x == 0.0:
        p = 1.0
    elif x == 1.0:
        p = x
    elif x > 1.0:
        nm1 = n-1
        p1 = ((2.0 * nm1 + 1.0)/n) * n * Legendre(nm1, x)
        p2 = (nm1 / n) * Legendre(n-2.0, x)
        p = p1 - p2
    return p

def boxmuller(points):
    np.random.seed(174)
    u1 = np.random.uniform(points)
    u2 = np.random.uniform(points)
    theta = 2.0*np.pi*u2
    rho = np.sqrt(-2*np.log(u1)) 
    x1 = rho * np.cos(theta) * u2
    x2 = rho * np.sin(theta) * u2
    return (x1, x2)

# def xorshift7(init):
#     x = np.array(8)
#     for i in range(8):
#         x[i] = init[i]
#     k = 0
# 	t = x[(k+7) and 0x7]
#     t = t ^ (t<<13)
#     y = t ^ (t<<9)
# 	t = x[(k+4) and 0x7]
#     y^= t ^ (t<<7)
# 	t = x[(k+3) and 0x7]
#     y^= t ^ (t>>3)
# 	t = x[(k+1) and 0x7]
#     y^= t ^ (t>>10)
# 	t = x[k]
#     t = t ^ (t>>7)
#     y^= t ^ (t<<24)
# 	x[k] = y
#     k = (k+1) & 0x7
# 	return np.float128(y * 2.32830643653869628906e-10)

if __name__ == '__main__':
    print("ID : ", os.getpid())
#    getcontext().prec = 8

    print('------------------------------------------------\n')
    # np.sqrt(n)/n = 1 / np.sqrt(n)
    print("Racines de f(x) = x^2 + 2x +1 : ", Quadratique(1, 2, 1)) # OK
    print("Racines de f(x) = x^2 + x - 2 : ", Quadratique(1, 1, -2)) # OK -2, 1
    print("Racines de f(x) = 4000x^2 -3999x - 1 : ", Quadratique(4000, -3999, -1)) # OK -0.00025, 1
    print('Racines de f(x) = -4x^2+4x : ', Quadratique(-4,4,0)) # OK 0, 1
    print('Racines de f(x) = 9x^2-9x : ', Quadratique(9,-9,0)) # OK 0, 1
    print('Racines de f(x) = -2x^2-2x : ', Quadratique(-2,-2,0)) # OK -1, 0
    print('Racines de f(x) = 6x^2+6x : ', Quadratique(6,6,0)) # OK -1, 0
    print('Racines de f(x) = -8x^2+x : ', Quadratique(-8,1,0)) # OK 0, 1/8=0.125
    print('Racines de f(x) = x^2+5x : ', Quadratique(1,5,0)) # OK -5, 0
    print('Racines de f(x) = x^2-4x : ', Quadratique(1,-4,0)) # OK 0, 4
    print('Racines de f(x) = -x^2+5x : ', Quadratique(-1,5,0)) # OK 0, 5
    print('Racines de f(x) = -x^2-4x : ', Quadratique(-1,-4,0)) # OK -4, 0
    print('Racines de f(x) = 15x^2+20x :', Quadratique(15,20,0)) # OK -4/3=-1.333, 0
    print('Racines de f(x) = -4x^2-10x :', Quadratique(-4,-10,0)) # OK -2.5, 0
    print('Racines de f(x) = -x^2+2x+4 :', Quadratique(-1,2,4)) # OK -1.2361,3.2361
    print('Racines de f(x) = x^2-x :', Quadratique(1,-1,0)) # OK 0, 1
    print('Racines de f(x) = x^2+x :', Quadratique(1,1,0)) # OK -1, 0
    print('Racines de f(x) = -x^2-x :', Quadratique(-1,-1,0)) # OK -1, 0
    print('Racines de f(x) = -x^2+x :', Quadratique(-1,1,0)) # OK 0, 1
    print('Racines de f(x) = x^2-2 :', Quadratique(1,0,-2)) # OK -np.sqrt(2),np.sqrt(2)
    print('Racines de f(x) = -x^2-1 :', Quadratique(-1,0,-1)) # OK -i,i
    print('Racines de f(x) = x^2+2 :', Quadratique(1,0,2)) # OK -np.sqrt(2), +np.sqrt(2)
    print('Racines de f(x) = -x^2+1 :', Quadratique(-1,0,1)) # OK -1, 1
    print('Racines de f(x) = -7x^2-6 :', Quadratique(-7,0,-6)) # OK -0.92582i, 0.92582i
    print('Racines de f(x) = 39x^2+10 :', Quadratique(39,0,10)) # OK -0.50637i, 0.50637i
    print('Racines de f(x) = -5x^2+32 :', Quadratique(-5,0,32)) # OK -2.5298, 2.5298
    print('Racines de f(x) = x^2+2x+1 :', Quadratique(1.0,2.0,1.0)) # OK -1
    print('Racines de f(x) = x^2-2x+1 :', Quadratique(1.0,-2.0,1.0)) # OK 1
    print('Racines de f(x) = 4x^2-5x :', Quadratique(4.0,-5.0,0.0)) # 0, 5/4=1.25
    print('Racines de f(x) = -2x^2+3x :', Quadratique(-2.0,3.0,0.0)) # 0, 3/2=1.5
    print('Racines de f(x) = -5x^2-x :', Quadratique(-5.0,-1.0,0.0)) # -1/5=-0.2, 0
    print('Racines de f(x) = 3x^2+2x :', Quadratique(3.0,2.0,0.0)) # -2/3=-0.666, 0
    print('Racines de f(x) = 3x^2-21x+30 :', Quadratique(3,-21,30)) # OK 2, 5
    print('Racines de f(x) = 3x^2+27x+60 :', Quadratique(3,27,60)) # OK -5, -4
    print('Racines de f(x) = x^2-6x+9 :', Quadratique(1,-6,9)) # OK 3
    print('Racines de f(x) = x^2-x-6 :', Quadratique(1,-1,-6)) # OK -2, 3
    print('Racines de f(x) = x^2-4x+13 :', Quadratique(1,-4,13)) # OK 2-3i, 2+3i
    print('Racines de f(x) = -4x^2+2x-8 :', Quadratique(-4,2,-8)) # OK 1/4-1.3919i, 1/4+1.3919i
    print('Racines de f(x) = 3x^2-24x+50 :', Quadratique(3,-24,50)) # OK 4-0.8164i, 4+0.8164i
    print('Racines de f(x) = x^2+3x+4 :', Quadratique(1,3,4)) # OK -1.5-1.3229i / 1.5000+i1.3228i
    print('Racines de f(x) = -5x^2+2x+3 :', Quadratique(-5,2,3)) # OK -0.6, 1
    print('Racines de f(x) = -2323x^2-53443 :', Quadratique(-2323,0,-53443)) # OK -4.7965i, 4.7965i
    print('Racines de f(x) = x^2-3.9999999999x+2.9999999999 :', Quadratique(1,-3.9999999999,2.9999999999)) # OK 1, 2.9999999999
    print('Racines de f(x) = x^2+2^28+2^54+1.0 :', Quadratique(1,2**28,2**54+1.0)) # OK -134217728.0 ??
    print('Racines de f(x) = 4.098933x^2-0.008735x :', Quadratique(4.098933,-0.008735,0.0)) # OK 0.0, 0.002131042395667360513
    print('Racines de f(x) = -0.023512x^2+3908768.989x :', Quadratique(-0.023512,3908768.989,0.0)) # OK 0.0, 166245703.85335147381
    print('Racines de f(x) = -51689000.1x^2-31809890.00009x :', Quadratique(-51689000.1,-31809890.00009,0.0)) # OK -0.61540927351175434, 0.0
    print('Racines de f(x) = 3.0090909873x^2+2090900009.7871x :', Quadratique(3.0090909873,2090900009.7871,0.0)) # OK -694861012.3827544451, 0.0
    print('Racines de f(x) = 100011.00003957x^2+1549x-0.3209 :', Quadratique(100011.00003957, 1549.0, -0.3209)) # OK -0.0152783, -0.000210014
    print("Racines de f(x) = x^2 + np.exp(5) * np.sqrt(8)x - atan(1/3) : ", Quadratique(1.0, exp(5) * np.sqrt(8), -atan(1/3))) # OK 419.776571355495, 0.000766480495463769
    print("Racines de f(x) = 96844x^2 - np.cos(2)log(2)x - np.sqrt(1/5)  : ", Quadratique(96844.0, -cos(2) * np.log(2), -np.sqrt(1/5))) # OK -0.00215041, 0.00214744
    print("Racines de f(x) = -4.0000000098*np.tan(1.0/3.0)x^2 + sqrt(sin(89))x - np.pi  : ", Quadratique(-4.0000000098*np.tan(1.0/3.0), np.sqrt(np.sin(89)), -np.pi)) # OK -1.20803, 1.87761
    print("Racines de f(x) = (np.sqrt(300020019.99001)/log(13))x^2 + 5.1*10e-9x + np.tan(sin(24)) : ", Quadratique(np.sqrt(300020019.99001)/log(13), 5.1*10e-6, tan(sin(24)))) # OK -0.0137393, 0.0137393
    print("Racines de f(x) = -0.2078964*np.sqrt(2**16)-np.log(2**32), 9**7*atan(4.0), 765350849021.01*np.sin(1.0/np.sqrt(5.0)) : ", Quadratique(-0.2078964*np.sqrt(2**16)-np.log(2**32), 9**7*atan(4.0), 765350849021.01*np.sin(1.0/np.sqrt(5.0)))) # OK (Decimal('-36421.1156329433'), Decimal('120521.3909460901'))
    print("Racines de f(x) = -4.000098^7*np.tan(3)-sin(log(7))x^2 + 12983.000000012+np.sqrt(sin(1/89))x + 190076.8931*np.pi/sin(cos(np.sqrt(7))) : ", Quadratique(-4.000098**7*np.tan(3)-sin(log(7)), 12983.000000012+np.sqrt(sin(1/89)), 190076.8931*np.pi/sin(cos(np.sqrt(7))))) # OK (Decimal('15.6497303554'), Decimal('-21.2100597867'))
    print('Racines de f(x) = atan(367/19)x^2-np.log(np.sqrt(13))x+4963*np.sin(1.1) :', Quadratique(atan(367/19),-np.log(np.sqrt(13)), 4963*np.sin(1.1))) # OK 0.422125-53.9584i, 0.422125-53.9584i
    print('Racines de f(x) = 97340028501.18964x^2 - (7^2^3*np.sqrt(35684))x - 1.00000006 : ', Quadratique(97340028501.18964, -(7**2)**3*np.sqrt(35684), -1.00000006)) # OK 0.011187412511526763, -9.18287991e-10
    print('Racines de f(x) = 8.05x^2 - 4338708.5x + 584608430061.25 :', Quadratique(8.05, -4338708.5, 584608430061.25)) # OK 269485-0.00253181377441101 i, 269485 + 0.00253181377441101 i
    # (8.05*x^2 - 4338708.5*x + 584608430061.25).roots(x, ring=CC)

    print('\n------------------------------------------------\n')

    print('Racines de f(x) = x^3+17x^2+63x-81 : ', Cubique(1,17,63,-81)) # -9, -9, 1
    print('Racines de f(x) = x^3-13x^2+16x+192 : ', Cubique(1,-13,16,192)) # -3, 8, 8
    print('Racines de f(x) = x^3-3x^2+3x-1 : ', Cubique(1,-3,3,-1)) # 1
    print('Racines de f(x) = x^3-5x^2+8x-4 : ', Cubique(1,-5,8,-4)) # 1, 2
    print('Racines de f(x) = x^3+6x-20 : ', Cubique(1,0,6,-20)) # 2, -1-3i, -1+3i
    print('Racines de f(x) = x^3-x-1 : ', Cubique(1,0,-1,-1)) # 1.3247, -0.6623-0.5622i, -0.6623+0.5622i
    print('Racines de f(x) = x^3+x^2+x-39 : ', Cubique(1,1,1,-39)) # 3, -2-3i, -2+3i
    print('Racines de f(x) = x^3-6x^2+34x-104 : ', Cubique(1,-6,34,-104)) # 4, 1-5i, 1+5i
    print('Racines de f(x) = x^3-7x+6 : ', Cubique(1,0,-7,6)) # -3, 1, 2
    print('Racines de f(x) = x^3-6x-301 : ', Cubique(1,0,-6,-301)) # 7.0, -3.5+5.5452i, -3.5-5.5452i
    print('Racines de f(x) = x^3-4x+3 : ', Cubique(1,0,-4,3)) # 1, 1.3027, -2.3027
    print('Racines de f(x) = x^3-4x+5 : ', Cubique(1,0,-4,5)) # -2.4567, 1.2283-0.7256i, 1.2283+0.7256i
    print('Racines de f(x) = x^3-6x+9 : ', Cubique(1,0,-6,9)) # -3, 1.5-0.866i, 1.5+0.866i
    print('Racines de f(x) = 2x^3-5x^2-1x+6 : ', Cubique(2,-5,-1,6)) # 2, -1, 3/2
    print('Racines de f(x) = 2x^3-10x^2+34x-26 : ', Cubique(2,-10,34,-26)) # 1, 2-3i, 2+3i
    print('Racines de f(x) = 7x^3-9x^2+10x+1 : ', Cubique(7,-9,10,1)) # -0.091863, 0.68879-1.03956i, 0.68879+1.03956i
    print('Racines de f(x) = 6x^3-72x^2+138x+216 : ', Cubique(6,-72,138,216)) # -1, 4, 9
    print('Racines de f(x) = 2x^3-12x^2-2x+60 : ', Cubique(2,-12,-2,60)) # 5, 3, -2
    print('Racines de f(x) = x^3+7x^2+11x+5 : ', Cubique(1,7,11,5)) # -5, (-1, -1)
    print('Racines de f(x) = 4x^3-40x^2-28x+784 : ', Cubique(4,-40,-28,784)) # -4, 7
    print('Racines de f(x) = 3x^3-54x^2+324x-648 : ', Cubique(3,-54,324,-648)) # 6
    print('Racines de f(x) = x^3+2x^2+20x+9 : ', Cubique(1,2,20,9)) # 0.4667, -0.767-4.324j, -0.767+4.324j, 
    print('Racines de f(x) = x^3-3x^2+6x-4 : ', Cubique(1,-3,6,-4)) # 1, 1 + i np.sqrt(3), 1 - i np.sqrt(3)
    print('Racines de f(x) = x^3-8x+8 : ', Cubique(1,0,-8,8)) # 2, -1-np.sqrt(5), np.sqrt(5)-1
    print('Racines de f(x) = -7x^3-19x^2+57x+101 : ', Cubique(-7,-19,57,101)) # -3.8556, -1.4462, 2.5876
    print('Racines de f(x) = 2x^3-4x^2+6x-4 : ', Cubique(2,-4,6,-4)) # 1, 0.5+1.3228i, 0.5-1.3228i
    print('Racines de f(x) = -13x^3+6x^2+9x-2 : ', Cubique(-13,6,9,-2)) # 1, -0.7449, 0.2065
    print('Racines de f(x) = 9x^3+x-10 : ', Cubique(9,0,1,-10)) # 1.0, -0.5+0.92796i, -0.5-0.9279i
    print('Racines de f(x) = x^3+x^2-2 : ', Cubique(1,1,0,-2)) #  1, -1-i, -1+i
    print('Racines de f(x) = x^3-300x+3000 : ', Cubique(1,0,-300.0,3000)) # w
    print('Racines de f(x) = x^3-x^2+x-1 : ', Cubique(1,-1,1,-1)) # 
    print('Racines de f(x) = -x^3+x^2-x+1 : ', Cubique(-1,1,-1,1)) # 
    print('Racines de f(x) = 34.0907x^3 + 189904.1x^2 + 0.3187x - 0.0000343771 : ', Cubique(34.0907,189904.1,0.3187,-0.0000343771)) # 
    print('Racines de f(x) = 34.9x^3 + 189904.1x^2 + 0.3x - 0.343771 : ', Cubique(34.9,189904.1,0.3,-0.343771)) # 
    print('Racines de f(x) = 4x^3 - 3x^2 - 5x + 2 : ', Cubique(4, -3, -5, 2)) #
    print('Racines de f(x) = x^3-np.exp(3^9/4^7)*x^2',end='')
    print('+20*np.tan((8/3)*np.pi^2*np.sqrt(5))*x',end='')
    print('+10^4*(log(4^6))/(np.sqrt(2)) : ', Cubique(1, \
            -np.exp(3**9/4**7), 20.0*np.tan((8/3)*np.pi**2*np.sqrt(5)), \
            10**4*np.log(4**6)/(np.sqrt(2))))
# WOLF -37.9976861421925288702967947...
# MOI  -37.997686142196858833, 20.661152403857848472-33.481144674398062543j, 20.661152403857848472+33.481144674398062543j
# SaGE -37.9976861421925,      20.6611524038557     -33.4811446744018j,      20.6611524038557     +33.4811446744018j

    # (x-10^6+1)(x-10^6+2)(x+10^6+3) = x^3 - 999994x^2 - 1000005999989x + 999999999993000006 = 0
    print('Racines de f(x) = x^3 - 999994 x^2 - 1000005999989 x + 999999999993000006 = 0 : ', Cubique(1, -999994, -1000005999989, 999999999993000006))

    x1, x2, x3 = Cubique2(1495, 1, -0.0000064287, 7513869)
    print(f'Racines de f(x) = 1495*x^3+x^2-0.0000064287*x+7513869 : \n{x1}\n{x2}\n{x3}')

    # MOI
    # -17.1295700280116
    # 8.56445056584526 + 14.834449706427j
    # 8.56445056584526 - 14.834449706427j

    # SaGE
    # -17.1295700309975, 1
    # 8.56445056733822 - 14.8344497038411*I, 1
    # 8.56445056733822 + 14.8344497038411*I, 1

    z1, z2, z3 = Cubique2(1, -9, 33, -65)
    print(f"Racines de f(x) = x^3-9x^2+33x-65 : \n{z1}\n{z2}\n{z3}")


    # print('\n------------------------------------------------\n')

    # print('Racines de f(x)=x^4-14x^3+71x^2-154x+120 : ', Quartic(1,-14,71,-154,120)) # 2, 3, 5, 4
    # print('Racines de f(x)=x^4-8x^3+18x^2-16x+5 : ', Quartic(1,-8,18,-16,5)) # !!! 5, 1, 1, 1
    # print('Racines de f(x)=x^4-2x^3-3x^2+4x+4 : ', Quartic(1,-2,-3,4,4)) # 2, 2, -1, -1
    # print('Racines de f(x)=x^4-4x^3+6x^2-4x+1 : ', Quartic(1,-4,6,-4,4)) # 1
    # print('Racines de f(x)=x^4-5x^3+13x^2-19x+10 : ', Quartic(1,-5,13,-19,10)) # 2, 1, 1+2i, 1-2i
    # print('Racines de f(x)=x^4-6x^3+26x^2-56x+80 : ', Quartic(1,-6,26,-56,80)) # 2+2i, 2-2i, 1-3i, 1+3i
    # print('Racines de f(x)=x^4-6x^2-x+1 : ', Quartic(8,-2,-9,2,1)) # -2.3202, -0.5137, 0.33587, 2.4980
    # print('Racines de f(x)=x^4 - (43 x^3)/40 + (63 x^2)/160 - (19 x)/320 + 1/320 = 0 : ', Quartic(1,-43/40,63/160,-19/320,1/320)) # 1/8, 1/5, 1/4, 1/2
    # print('Racines de f(x)=2x^4+7x^3-35x^2+x+1 : ', Quartic(524288/32768,-3639296/32768,-215184/32768,1007/32768,7/32768)) # -1/16=-0.0625, -1/256=-00390625, 1/128=0.0078125, 7

    print("------------------------------------------------")

    # x^45-x^x^(x-1)+np.log(x^2-x+3)-atan(2pi+x)+10^30=0 > -4,6...

    f_Dichotomie = lambda x: x**3-12*x**2+48*x-64
    print('Dichotomie(x^3-12x^2+48x-64) :', Dichotomie(f_Dichotomie, 3.7, 5.2, 20)) #x=4

    print("------------------------------------------------")

    # x^45-x^x^(x-1)+np.log(x^2-x+3)-atan(2pi+x)+10^30=0 > -4,6...

    # f_Quadrisection = lambda x: x**45-x**(x**(x-1.0))+np.log(x**2-x+3.0)-atan(x+2*np.pi)+10**30
    # f_Quadrisection = lambda x: x**3-12*x**2+48*x-64
    # print('Quadrisection(x^3-12x^2+48x-64), root: ', Quadrisection(f_Quadrisection, 2, 6, 20, 1e-4))

    print("------------------------------------------------")

    f = lambda x: 5*x**4-x**3+7*x**2
    print('df=',derivative1(f, 4))
    print('d2f=',derivative2(f, 4))
    print('d3f=',derivative3(f, 4))

    print("------------------------------------------------")

    f_Secante = lambda x: 7*x**5 - 105*x**4 + 630*x**3 - 1890*x**2 + 2835*x - np.sin(x) - 1701 #3,1411997663343662253
    print('secante(fsecante, 2, 4, 100, 1e-8), root: ', Secante(f_Secante, 1, 5, 10, 1e-8))

    print("------------------------------------------------")

    f_RegulaFalsi = lambda x: 7*x**5 - 105*x**4 + 630*x**3 - 1890*x**2 + 2835*x - np.sin(x) - 1701 #3,1411997663343662253
    print('RegulaFalsi(), root: ', RegulaFalsi(f_RegulaFalsi, 1, 5, 10, 1e-8))

    print("------------------------------------------------")

    # x ≈ 0.726914253804356...
    # f_Newton = lambda x: x**23 + x**9 *np.sin(x)*np.exp(x) - 1
    # df_newton = lambda x: 23*x**22 + np.exp(x)*(x+1)*np.sin(x) + np.exp(x)*x*np.cos(x)
    # print('NewtonRaphson(x^23 + x*np.sin(x)*np.exp(x) - 1), x =', NewtonRaphson(f_Newton, df_newton, 1.0, 10))
    
    # x≈-1.1662383633886558404
    f_Newton = lambda x: x**11-2*np.cos(x**7-x**6+x**2-1)-x+5
    df_newton = lambda x: 11*x**10 + 2*(7*x**5 - 6*x**4 + 2) * x * np.sin(x**7 - x**6 + x**2 - 1) - 1
    print("NewtonRaphson(x^11-2*cos(x^7-x^6+x^2-1)-x+5), x =", NewtonRaphson(f_Newton, df_newton, -1.2, 10))

    print("------------------------------------------------")

    #0.626884
    f_Halley = lambda x: x**11+x**3*np.exp(x)+np.log(x)
    df_Halley = lambda x: 11*x**10+x**2*np.exp(x)*(x+3)+(1/x)
    df2_Halley = lambda x: 110*x**9 + x*np.exp(x)*(x**2 + 6*x + 6) - (1/x**2)
    print('Halley(x**11+x**3*np.exp(x)+np.log(x)), root: ', Halley(f_Halley, df_Halley, df2_Halley, 0.5))

    print("------------------------------------------------")

    # x ≈ 0.926753225519136...
    f_Steffesen = lambda x: x**25+x+np.log(x)-1
    print('Steffesen(x^25 + x + np.log(x) - 1), x =', Steffesen(f_Steffesen, 0.9))

    print("------------------------------------------------")

    #x ≈ 0.882452014582186...
    f_Li = lambda x: x**14 * np.log(10.0 * x**13)+ x - 1.0
    df_Li = lambda x: 13.0 * x**13 + 14.0*x**13*np.log(10.0*x**13) + 1.0
    print('Li(x^14 * np.log(10 * x^13)+ x - 1), root: ', Li(f_Li, df_Li, 0.9, 10))

    print("------------------------------------------------")

    # x ≈ -1.15566279371551
    f_PiscoranMiclaus = lambda x: x**11 + np.pi/(x**2) - np.log(x**2 - x + 9.0) + 5.0
    df_PiscoranMiclaus = lambda x: 11.0*x**10 - (2.0*np.pi)/(x**3) + (1.0-2.0*x)/(x**2-x+9.0)
    print('PiscoranMiclaus(x^11 + np.pi/(x^2) - np.log(x^2 - x + 9.0) + 5.0), x =', PiscoranMiclaus(f_PiscoranMiclaus, df_PiscoranMiclaus, -1.2))

    print("------------------------------------------------")

    # x -1.10983360581301...
    f_NoorKhanHussain = lambda x: x**9-x**3*np.log(x**2)-cos(4*x)+2
    df_NoorKhanHussain = lambda x: 9*x**8 - 3*x**2*np.log(x**2) - 2*x**2 + 4*np.sin(4*x)
    print('NoorKhanHussain(x^9-x^3*np.log(x^2)-cos(4*x)+2), x =', NoorKhanHussain(f_NoorKhanHussain,df_NoorKhanHussain,-1.0))

    print("------------------------------------------------")

    # x ≈ 0.872709679318703572407316240...
    #     0.872709679318703613
    f_XiaofengDongweiDongyang = lambda x: x**5+np.sin(x)+np.log(x**2)-1.0
    df_XiaofengDongweiDongyang = lambda x: 5*x**4+np.cos(x)+2.0/x
    print('XiaofengDongweiDongyang, x =', XiaofengDongweiDongyang(f_XiaofengDongweiDongyang,df_XiaofengDongweiDongyang, 0.8))

    print("------------------------------------------------")

    # x ≈ 0.957026793632555524260286372...
    f_MuhaijirSolehSafitri = lambda x: x**5*np.sqrt(x**3) - np.sin(x*np.pi)+x*np.exp(-x*x)-1.0
    df_MuhaijirSolehSafitri = lambda x: -2.0*x**2*np.exp(-x**2) + np.exp(-x**2) + (3.0*x**7)/(2.0**np.sqrt(x**3)) + 5.0**np.sqrt(x**3)*x**4-pi*np.cos(pi*x)
    print('MuhaijirSolehSafitri(np.sqrt(x^5) - np.sin(x*np.pi)+np.exp(-x^2)*x-1), root: ', MuhaijirSolehSafitri(f_MuhaijirSolehSafitri, df_MuhaijirSolehSafitri, 0.9, 10))

    print("------------------------------------------------")

    # x ≈ 0.828104183060519108169723386...
    f_CorderoHuesoMartinezTorregrosa = lambda x: 8.0 * x**5 - np.sin(x)**2 - x*atan(x) - 2.0
    print('CorderoHuesoMartinezTorregrosa(8.0 * x^5 - np.sin(x)^2 - x*atan(x) - 2.0), x =', CorderoHuesoMartinezTorregrosa(f_CorderoHuesoMartinezTorregrosa,0.82))

    print("------------------------------------------------")

    # x ≈ 2.29694748561318151142281970...
    f_JunjuaZafarYasmin_MNP16= lambda x: x**3 - np.log(x**2+x-2.0) + np.sin(x*np.sqrt(np.pi))/2.0-10.0
    print('JunjuaZafarYasmin_MNP16(), x =', JunjuaZafarYasmin_MNP16(f_JunjuaZafarYasmin_MNP16, 2.2))

    print("------------------------------------------------")

    # x ≈ 1.11428262886646158733693110...
    #     1.114282628866461566
    f_EdwarImranDeswita = lambda x: exp(-x)-x**3+np.sqrt(x*np.sin(x**4))
    print('EdwarImranDeswita(exp(-x) - x^3 + np.sqrt(x*np.sin(x^4))), x =', EdwarImranDeswita(f_EdwarImranDeswita, 0.8))

    print("------------------------------------------------")

    # x ≈ 1.30327829147035
    f_KantaloMuangchanSompong = lambda x: x**5 * np.log(x) + np.exp(x**4-x**3-8) - 1.0
    print('KantaloMuangchanSompong(x^5 * np.log(x) + np.exp(x^4-x^3-8) - 1.0), x =', KantaloMuangchanSompong(f_KantaloMuangchanSompong, 1.3, 10, 1e-8))

    print("------------------------------------------------")
    # x ≈ 1.68073513421261...
    f_LotfiEftekhari = lambda x: x**5+np.log(x**3)-np.sqrt(2)*x-2.0*np.sqrt(x)-10.0
    df_LotfiEftekhari = lambda x: -(-5.0*x**5 + x*np.sqrt(2) + np.sqrt(x) - 3.0)/x
    print('LotfiEftekhari(x^3-cos(x^2-1)*atan(x)-x-2), x =', LotfiEftekhari_equ_27(f_LotfiEftekhari, df_LotfiEftekhari, 1.6))

    print("------------------------------------------------")

    # x ≈ 1.18886347689168...
    f_SharmaBahl = lambda x: x**9-pi*x**3-atan(x)+2*x*np.log(x+3)-2
    df_SharmaBahl = lambda x: 9*x**8 - 3*np.pi*x**2 - (1/(x**2+1)) + ((2*x)/(x+3)) + 2*np.log(x+3)
    print('SharmaBahl(x^9 - np.pi*x^3 - atan(x) + 2*x*np.log(x+3) - 2), x =', SharmaBahl(f_SharmaBahl,df_SharmaBahl, 1.1, 10, 1e-8))

    print("------------------------------------------------")
    #x ≈ 0.450115674711598...
    f_soleymani2 = lambda x: x**25+np.sin(2.0*x-1)-cos(x)+1.0
    # x ≈ 1.14568010950824...
    # f_soleymani2 = lambda x: x**11-x*np.exp(-x**2-1.0)+4.0*np.cos(x)+6.0
    # print('soleymani2(x^11-x*np.exp(-x^2-1)+4.0*np.cos(x)-6), x =', Soleymani2(f_soleymani2, 0.45))

    # x ≈ 0.769510671273811...
    f_Soleymani = lambda x: x**11-x*np.exp(4.0*x**2-1.0)+np.sqrt(1.0+4.0*(cos(x)))+1.0
    df_Soleymani = lambda x: 11.0* np.power(x,10)-8.0*np.exp(4.0*x*x-1.0)*x*x-np.exp(4.0*x*x-1.0)-np.sqrt(2.0*np.sin(x))/np.sqrt(1.0+4.0*np.cos(x))
    print('Soleymani(x^11 - x*np.exp(4.0 * x^2 - 1.0)+np.sqrt(1.0 + 4.0 * (cos(x))) + 1.0), x =', Soleymani(f_Soleymani, df_Soleymani, 0.7, 10, 1e-8))

    print("------------------------------------------------")

    # x ≈ 1.27311032510007475756096131...
    f_TaoMadhu = lambda x: x**11 - 4 * x * np.log(x**5) - np.sqrt(x + 2) - 2 * np.pi
    df_TaoMadhu = lambda x: 11 * x**10 - 4 * np.log(x**5) - 1 / (2 * np.sqrt(x + 2)) - 20
    print('TaoMadhu(x^11 - 4xlog(x^5) - sqrt(x + 2) - 2pi), x =', TaoMadhu(f_TaoMadhu, df_TaoMadhu, 1.2))

    print("------------------------------------------------")

    # x ≈ 1.26613545059512...
    f_RalevicCebic_alg1 = lambda x: x**11-(2.0*np.pi)/np.sqrt(x**2)+np.log(x**2+pi)-10.0
    df_RalevicCebic_alg1 = lambda x: x*(11.0*x**9 + 2.0/(x**2+pi)+(2.0*np.pi)/((x**2)**(3.0/2.0)))
    print("RalevicCebic_alg1(), root:", RalevicCebic_alg1(f_RalevicCebic_alg1, df_RalevicCebic_alg1, 1.2))

    print("------------------------------------------------")

    #x ≈ -1.04855859799654...
    f_Jaiswal = lambda x: x**11-x**7*np.cos(x)+np.exp(2*x)-sin(x)
    print('Jaiswal(x**11-x**7*np.cos(x)+np.exp(2*x)-sin(x)), x =', Jaiswal(f_Jaiswal, -1.0))

    print("------------------------------------------------")

    # x ≈ -0.00664881044259086...
    f_tdsm = lambda x: 3*x**5-15*x*np.log(x**2)+np.cos(x)
    print('tdsm(3*x**5-15*x*np.log(x**2)+np.cos(x)), x =', tdsm(f_tdsm, -0.1))

    print("------------------------------------------------")

    # x≈2.19488
    f_SolaimanHashim_equ17 = lambda x: x**5 - np.cos(pi*np.exp(-x)) - 50.0
    df_SolaimanHashim_equ17 = lambda x: 5.0*x**4 - np.pi*np.exp(-x) * sin(pi*np.exp(-x))
    print('SolaimanHashim_equ17(x^5 - np.cos(pi*np.exp(-x)) - 50), x =', SolaimanHashim_equ17(f_SolaimanHashim_equ17, df_SolaimanHashim_equ17, 2.1))

    print("------------------------------------------------")

    # x ≈ 1.54926807995873669402890616...
    # f_JanngamaComemuangEqu23 = lambda x: x**3-cos(x**2-1.0)-atan(x)-np.sqrt(x)-2.0
    # df_JanngamaComemuangEqu23 = lambda x: 3.0*x**2 + 2.0*x*np.sin(x**2-1.0) - 1.0/(x**2+1.0) - 1.0/(2.0*np.sqrt(x))
    # print('JanngamaComemuangEqu23() = :', JanngamaComemuangEqu23(f_JanngamaComemuangEqu23, df_JanngamaComemuangEqu23, 1.5, 10, 1e-8))

    # print("------------------------------------------------")

    #x ≈ 0.726914253804356...
    f_RafiullahJabeen = lambda x: x**23 + x * sin(x) * np.exp(x) - 1.0
    df_RafiullahJabeen = lambda x: 23 * x**22 + np.exp(x) * (x+1.0) * sin(x) + np.exp(x)*x*np.cos(x)
    print('RafiullahJabeen(x^17-np.exp(x)+x^x-1), x =', RafiullahJabeen(f_RafiullahJabeen, df_RafiullahJabeen, 0.7))

    print("------------------------------------------------")

    # x ≈ 1.82391763463176311794936818...
    # f_Matinfaraminzadeh_eq21 = lambda x: x**3-np.exp(-x)-np.log(x**2-x+7) - x*np.sin(x)-2.0
    # alpha = -2.0
    # print('Matinfaraminzadeh_eq21(), x =', Matinfaraminzadeh_eq21(f_Matinfaraminzadeh_eq21, 1.8, alpha, 10, 1e-8))

    # print("------------------------------------------------")

    # x ≈ -0.913230718914925836843523310...
    f_LiWangMadhu_PM16 = lambda x: x**2-np.exp(x) + x*np.cos(x**3*np.sqrt(2.0))
    # f_LiWangMadhu_PM16 = lambda x: 5*x**4 - np.log(x**2) + 0.5 / pow((x+2.0),3.0/2.0) - 2.0
    print('LiWangMadhu_PM16(), x =', LiWangMadhu_PM16(f_LiWangMadhu_PM16, -0.9))

    print("------------------------------------------------")

    #x ≈ 1.30361055802424...
    f_WangLiu = lambda x: x**14+4.0*x**2-x*np.exp(x**7-2.0*x)+10.0
    df_WangLiu = lambda x: 14.0*x**13 + 8.0*x - exp(x**7-2.0*x)*(7*x**6-2.0)
    print('WangLiu(x**14+4*x**2-x*np.exp(x**7-2*x)+10), x =', WangLiu(f_WangLiu, df_WangLiu, 1.1))

    print("------------------------------------------------")

    # x≈2.19488...
    f_SolaimanHashim = lambda x: pow(x,5) - np.cos(pi*np.exp(-x)) - 50.0
    df_SolaimanHashim = lambda x: 5.0* np.power(x,4) - np.pi*np.exp(-x) * sin(pi*np.exp(-x))
    print('SolaimanHashim(x^5 - np.cos(pi*np.exp(-x)) - 50.0, x =', SolaimanHashim(f_SolaimanHashim, df_SolaimanHashim, 2.2))

    print("------------------------------------------------")

    # x≈1.0817128340146981229
    f_SolaimanHashim2 = lambda x: x**15 - x**7 + np.cos(pi*np.exp(-x)) - 2.0
    df_SolaimanHashim2 = lambda x: (15.0*x**8 - 7.0)*x**6 + np.pi* np.exp(-x)* sin(pi*np.exp(-x))
    print('SolaimanHashim2(root:', SolaimanHashim2(f_SolaimanHashim2, df_SolaimanHashim2, 2.1))

    print("------------------------------------------------")

    # x ≈ 1.64305371855953491490026903...
    f_FitriyaniImranSyamsudhuha = lambda x: x**6 * np.power(np.log(x),2) + np.exp(x*np.sin(x)) - 10.0
    df_FitriyaniImranSyamsudhuha = lambda x: 6*x**5 * np.power(np.log(x),2) + 2.0*x**5*np.log(x) + np.exp(x*np.sin(x))*(np.sin(x) + x*np.cos(x))
    print('FitriyaniImranSyamsudhuha(x^6 * np.power(log(x),2) + np.exp(x*np.sin(x)) - 10.0), x =', FitriyaniImranSyamsudhuha(f_FitriyaniImranSyamsudhuha, df_FitriyaniImranSyamsudhuha, 1.64))

    print("------------------------------------------------")

    # x ≈ 1.06873197457578...
    f_GhaneKanafiKordrostami = lambda x: log(x) + x**2/np.sqrt(x) - np.cos(x/2.0)
    df_GhaneKanafiKordrostami = lambda x: x**2 / (2* np.power(1+x,3/2)) + (2*x)/(np.sqrt(1+x)) + 1.0/x + 0.5 * sin(0.5*x)
    print('GhaneKanafiKordrostami(log(x) + x^2/np.sqrt(x) - np.cos(x/2.0)), x =', GhaneKanafiKordrostami(f_GhaneKanafiKordrostami, df_GhaneKanafiKordrostami, 1.0, 10))

    print("------------------------------------------------")

    # x ≈ 1.21615028665551090098790271...
    f_Deepargyros = lambda x: x**11 - x*np.log(x) + np.arctan(x+0.5)-x**4*np.sqrt(x) - 7
    df_Deepargyros = lambda x: 11*x**10 - (9*x**(7/2))/2 + 1/((x + 0.5)**2 + 1) - np.log(x) - 1
    print('Deepargyros(x^11 - x*log(x) + atan(x+0.5)-x^4*sqrt(x)-7), x =', Deepargyros(f_Deepargyros, df_Deepargyros, 1.2, 10))

    print("------------------------------------------------")
    # x ≈ 1.2072039913516103323
    f_UllahalFhaidahmad = lambda x: x**19-(x**3)*np.cos(x)-4*np.pi*x-20
    df_UllahalFhaidahmad = lambda x: 19*x**18+x**3*np.sin(x)-3*(x**2)*np.cos(x)-4*np.pi
    print('UllahalFhaidahmad(x^19-x^3*np.cos(x)-4pi*x-20), x =', UllahalFhaidahmad(f_UllahalFhaidahmad, df_UllahalFhaidahmad, 1.2))

    print("------------------------------------------------")

    # x ≈ 1.02860739761363
    f_SabaNaseemSaleem = lambda x: pow(x,59) - exp(x) + np.cos(x) - 3.0
    df_SabaNaseemSaleem = lambda x: 59.0* np.power(x,58) - exp(x) - np.sin(x)
    print('SabaNaseemSaleem(x^59 - exp(x) + np.cos(x) - 3.0), x =', SabaNaseemSaleem(f_SabaNaseemSaleem, df_SabaNaseemSaleem, 1.2))

    print("------------------------------------------------")

    #x ≈ 0.317774066698266...
    f_ChunNeta = lambda x: x**7 + np.sqrt(pi*x) - np.cos(x**3)
    df_ChunNeta = lambda x: 7.0 * x**6 + 3.0*x**2 * sin(x**3) + np.sqrt(np.pi)/(2.0*np.sqrt(x))
    print('ChunNeta(x^7 + np.sqrt(pi*x) - np.cos(x^3)), x =', ChunNeta(f_ChunNeta, df_ChunNeta, 0.3))

    print("------------------------------------------------")

    # x ≈ 0.138693931574463
    f_SalimiNikLongSharifiPansera = lambda x: (x-2)**5-np.log(x+4.0) - np.sin((x*np.pi)/4.0) + 100.0*x + 10.0
    df_SalimiNikLongSharifiPansera = lambda x: 5.0*(x-2.0)**4 - 1.0/(x+4.0) - (1.0/4.0)*np.pi*np.cos((pi*x)/4.0) + 100.0
    print('SalimiNikLongSharifiPansera(x^7 + np.sqrt(pi*x) - np.cos(x^3)), x =', SalimiNikLongSharifiPansera(f_SalimiNikLongSharifiPansera, df_SalimiNikLongSharifiPansera, 0.1))

    print("------------------------------------------------")

    # x≈0.54402533713080976719
    f_SharifiSalimiSiegmundLotfi_equ_M7 = lambda x: x**7-np.sqrt(np.pi)*(2.0*x**2-4.0*x+2.0)+np.exp(x)-1.0
    df_SharifiSalimiSiegmundLotfi_equ_M7 = lambda x: 7.0*x**6+ np.exp(x) - 4.0*np.sqrt(np.pi)*(x-1.0)
    print('SharifiSalimiSiegmundLotfi_equ_M7(x^7-np.sqrt(pi)*(2*x^2-4*x+2)+np.exp(x)-1), x =', SharifiSalimiSiegmundLotfi_equ_M7(f_SharifiSalimiSiegmundLotfi_equ_M7, df_SharifiSalimiSiegmundLotfi_equ_M7, 0.5))

    print("------------------------------------------------")

    # 1.61962720543558
    f_KhattriSteihaug_equ_M8 = lambda x: 13.0*x**9-np.exp(x-1000.0)+np.arctan(4.0*x)+x-1000.0
    print('KhattriSteihaug_equ_M8(13x^9-np.exp(x-1000)+np.arctan(4.0*x)+x-1000), x =', KhattriSteihaug_equ_M8(f_KhattriSteihaug_equ_M8, 1.57))

    print("------------------------------------------------")

    # x≈1.0531660225496405131
    f_Thukral = lambda x: 64.0*x**5 - 32.0*np.sin(2.0*x) - 16.0*x*np.cos(2.0*x) - 64.0
    print('Thukral(64x^5 - 32sin(2x) - 16x.cos(2x)-64), x =', Thukral(f_Thukral, 1.0))
    
    print("------------------------------------------------")

    # x≈1.7011931868965984211
    f_BiRenWu_equ_3 = lambda x: x**6-np.exp(x**3)-1.0/np.sqrt(5.0+x)+8.0*x+100.0
    df_BiRenWu_equ_3 = lambda x: 6.0*x**5-3.0*np.exp(x**3)*x**2 + 1.0/(2*(5.0+x)**(3/2)) + 8.0
    print('BiRenWu_equ_3(x^5-np.exp(x^2+4)-np.log(x)+100), x =', BiRenWu_equ_3(f_BiRenWu_equ_3, df_BiRenWu_equ_3, 1.7))

    print("------------------------------------------------")

    # x ≈ 1.21951339224370...
    f_Bawazir = lambda x: pow(x,26) * np.power(log(x),2) + np.exp(x*np.sin(x)) - 10.0
    df_Bawazir = lambda x: 26.0* np.power(x,25) * np.power(log(x),2) + 2.0* np.power(x,25) * np.log(x) + np.exp(x*np.sin(x)) * (sin(x)+x*np.cos(x))
    print('Bawazir(x^26 * np.log(x)^2 + np.exp(x*np.sin(x)) - 10.0), x =', Bawazir(f_Bawazir, df_Bawazir, 1.2))

    print("------------------------------------------------")

    # x ≈ -1.00550855885479...
    f_QureshiBozdarPirzadaarain = lambda x: x**4 + np.sin(3*x) - np.cos(x)-np.exp(x**3)
    df_QureshiBozdarPirzadaarain = lambda x: (4*x-3*np.exp(x**3))*x**2 + np.sin(x)+3*np.cos(3*x)
    print('QureshiBozdarPirzadaarain(x^4 + np.sin(3x) - np.cos(x)-np.exp(x^3)), x =', QureshiBozdarPirzadaarain(f_QureshiBozdarPirzadaarain, df_QureshiBozdarPirzadaarain, -0.9))

    print("------------------------------------------------")

    # x ≈ 0.984385886271408...
    f_aliaslamalianwarNadeem = lambda x: x*x*x*np.exp(x**x)-cos(x)-2.0
    df_aliaslamalianwarNadeem = lambda x: exp(x*x)*(2*x*x + 3.0) * x*x + np.sin(x)
    d2f_aliaslamalianwarNadeem = lambda x: 2*np.exp(x*x)*x*(2*x*x*x*x + 7.0*x*x + 3.0) + np.cos(x)

    print('aliaslamalianwarNadeem(x^26 * np.log(x)^2 + np.exp(x*np.sin(x)) - 10.0), x =', aliaslamalianwarNadeem(f_aliaslamalianwarNadeem, df_aliaslamalianwarNadeem, d2f_aliaslamalianwarNadeem, 1.2))

    print("------------------------------------------------")

    # x ≈ 1.58447983918569...
    f_LiMuMaWang = lambda x: x**15 + np.exp(x) - np.sin(x**3-2.0*x+1.0)-10**3
    df_LiMuMaWang = lambda x: 15.0*x**14 - 3.0*(x**2-2.0/3.0)*np.cos(x**3-2.0*x+1.0) + np.exp(x)
    print('LiMuMaWang(x^5 + 4.0*np.log(x) + np.exp(x) - 1.0)), x =', LiMuMaWang(f_LiMuMaWang, df_LiMuMaWang, 1.5))

    print("------------------------------------------------")

    # x ≈ 1.56894579304308
    f_ZhangZhangDing = lambda x: np.exp(-x) - x**x + x*np.sin(x) + 0.25
    df_ZhangZhangDing = lambda x: x**x*(-(np.log(x) + 1.0)) + np.sin(x) + x*np.cos(x) - np.exp(-x)
    print('ZhangZhangDing(exp(-x) - x^x + x*np.sin(x) + 1/4), x =', ZhangZhangDing(f_ZhangZhangDing, df_ZhangZhangDing, 1.6))

    print("------------------------------------------------")

    # x ≈ 14.3098
    f_BiazarGhanbari_equ_13 = lambda x: x**5 + x**3 -np.exp(x-1.0) + 8.0*x + 1.0
    df_BiazarGhanbari_equ_13 = lambda x: 5.0*x**4 + 3.0*x**2 - exp(x-1.0) + 8.0
    print('BiazarGhanbari_equ_13(x^5 + x^3 -np.exp(x-1) + 8x + 1.0), x =', BiazarGhanbari_equ_13(f_BiazarGhanbari_equ_13, df_BiazarGhanbari_equ_13, 14.3))

    print("------------------------------------------------")


    # x ≈ 1.73276919897206...
    # f_KimBehlMotsaEq33 = lambda x: x**9 - exp(x+2)+(sin(x))**2-100
    # df_KimBehlMotsaEq33 = lambda x: 9*x**8 - exp(x+2) + 2*np.sin(x)*np.cos(x)
    f_KimBehlMotsaEq34 = lambda x: x**9 - exp(x+2)+(sin(x))**2-100
    df_KimBehlMotsaEq34 = lambda x: 9*x**8 - exp(x+2) + 2*np.sin(x)*np.cos(x)
    # print('KimBehlMotsaEq33(x^9 - exp(x+2)+(sin(x))^2-100), x =', KimBehlMotsaEq33(f_KimBehlMotsaEq33, df_KimBehlMotsaEq33, 1.7, 20, 1e-8))
    print('KimBehlMotsaEq34(x^9 - exp(x+2)+(sin(x))^2-100), x =', KimBehlMotsaEq34(f_KimBehlMotsaEq34, df_KimBehlMotsaEq34, 1.6))

    print("------------------------------------------------")

    #x ≈ 1.08964327252887...
    f_KumarSingh = lambda x: x**9-np.sqrt(10+x)+np.exp(-x)+atan(x)
    df_KumarSingh = lambda x: 9*x**8 + 1/(x**2+1) - exp(-x) - 1/(2*np.sqrt(10+x))
    print('KumarSingh(x^9-np.sqrt(10-x)+np.exp(-x)+atan(x)), x =', KumarSingh(f_KumarSingh, df_KumarSingh, 1.1))

    print("------------------------------------------------")

    # x ≈ 1.61350763512373435597073879...
    f_SoleymaniShateyiSalmaniEqu215 = lambda x: x**13-np.log(2.0*x-1.0)-(x*np.sqrt(5.0))+np.exp(x)*100.0-1000.0
    df_SoleymaniShateyiSalmaniEqu215 = lambda x: 13.0*x**12 + 100.0*np.exp(x) - 2.0/(2.0*x - 1.0) - np.sqrt(5.0)
    print('SoleymaniShateyiSalmani215(exp(x^2-4x+2)-(log(x))^2+atan(x)-4), x = ', SoleymaniShateyiSalmaniEqu215(f_SoleymaniShateyiSalmaniEqu215, df_SoleymaniShateyiSalmaniEqu215, 1.6))

    print("------------------------------------------------")

    # x ≈ 5.34690180838635...
    # f_akramZafarYasmin_MFM16 = lambda x: x**7+np.exp(2.0)**2+atan(x)-50.0**3
    # print('akramZafarYasmin_MFM16() :', akramZafarYasmin_MFM16(f_akramZafarYasmin_MFM16, 5.3, 10, 1e-8))
    # x ≈ 1.64305371855953491490026903...
    # f_akramZafarYasmin_MFM16 = lambda x: x**6 * np.power(log(x),2) + np.exp(x*np.sin(x)) - 10.0
    # print('akramZafarYasmin_MFM16() :', akramZafarYasmin_MFM16(f_akramZafarYasmin_MFM16, 1.7, 10, 1e-8))

    # print("------------------------------------------------")

    # 1.61962720543558
    # f_ZafarHussainFatimahKharalM2 = lambda x: 13*x**9-np.exp(x-1000.0)+atan(4.0*x)+x-1000.0
    # df_ZafarHussainFatimahKharalM2 = lambda x: 117.0*x**8 + 4.0/(16*x**2+1.0) - exp(x-1000.0) + 1.0
    # x ≈ 2.18451157378143...
    f_ZafarHussainFatimahKharalM2 = lambda x: x**3+np.exp(x)-np.log(x**2)+(1.0+x)/np.sqrt(2.0)-20.0
    df_ZafarHussainFatimahKharalM2 = lambda x: 3.0*x**2 + np.exp(x) - 2.0/x + 1.0/np.sqrt(2.0)
    print('ZafarHussainFatimahKharalM2((x-np.exp(2))^3+atan(10-x)+13.004), x =', ZafarHussainFatimahKharalM2(f_ZafarHussainFatimahKharalM2, df_ZafarHussainFatimahKharalM2, 2.0))

    print("------------------------------------------------")

    # x ≈ 2.06655323966041167704033691...
    f_SharmaargyrosKumar = lambda x: x**5 + np.sin(3.0*x) - x*np.cos(x)-4.0*np.exp(x)-7.0
    df_SharmaargyrosKumar = lambda x: 5.0*x**4 - 4.0*np.exp(x) + x*np.sin(x) - np.cos(x) + 3.0*np.cos(3.0*x)
    print('SharmaargyrosKumar(), x =', SharmaargyrosKumar(f_SharmaargyrosKumar, df_SharmaargyrosKumar, 2.0))

    print("------------------------------------------------")

    # x ≈ 1.23006617693139398830483636...
    f_GeumKimY1 = lambda x: 13*x**9 - x * np.exp(x**7) + np.cos(x) + np.sqrt(8)/x**3 + x**2
    df_GeumKimY1 = lambda x: 117*x**8 - np.exp(x**7)*(7*x**7 + 1) - (6*np.sqrt(2))/x**4 + 2*x - np.sin(x)
    print('GeumKimY1(X) =', GeumKimY1(f_GeumKimY1, df_GeumKimY1, 1.3))
    exit()
    print("------------------------------------------------")

    # x ≈ 0.456565842126133897198649539...
    f_EsmaeiliahmadiErfanifar = lambda x: x+np.exp(x)-x*np.sin(x-1.0/3.0)+x*np.sqrt(5.0)-3.0
    df_EsmaeiliahmadiErfanifar = lambda x: exp(x)-sin(x-1.0/3.0) - x*np.cos(x-1.0/3.0) + np.sqrt(5.0) + 1.0
    print('EsmaeiliahmadiErfanifar, x =', EsmaeiliahmadiErfanifar(f_EsmaeiliahmadiErfanifar, df_EsmaeiliahmadiErfanifar, 0.4))

    print("------------------------------------------------")

    # x ≈ 16.6171
    f_saeed = lambda x: np.sqrt(x+1.0) - exp(x) + 13.0*x**5-1.0 + 1.0/x
    df_saeed = lambda x: 65.0*x**4 - 1.0/x**2 - exp(x) + 1.0 / (2.0*np.sqrt(x+1.0))
    print('Saeed(np.sqrt(x+1) - exp(x) +13*x**5-1 + 1/x), x =', Saeed(f_saeed, df_saeed, 16.5, 10, 1e-8))

    print("------------------------------------------------")

    # x≈2.03991
    f_NaseemRehmanYounis = lambda x: x**3 + x**2/np.sqrt(5.0)+np.sin(2.0*x)**2 - 11.0
    print('NaseemRehmanYounis(x^3 + x^2/np.sqrt(5)+np.sin(2x)^2 - 11.0), x =', NaseemRehmanYounis(f_NaseemRehmanYounis, 2.0))

    print("------------------------------------------------")
    # x≈-1.06528
    f_Com69 = lambda x: x**23 - np.sin(2.0*x)**2 + 5.0
    df_Com69 = lambda x: 23*x**22 - 4.0*np.cos(2.0*x)*np.sin(2.0*x)
    print('Com69( x^23 - np.sin(2x)^2 + 5.0), x =', Com69(f_Com69, df_Com69, -1.1))

    print("------------------------------------------------")

    # x ≈ 1.38250306834130503627689470...
    f_SanaaslamNoorInayatNoor = lambda x: x**9 - np.pi*np.sin(x) - np.log(2.0+x**2) - 14.0
    df_SanaaslamNoorInayatNoor = lambda x: 9.0*x**8 - (2.0*x)/(2.0+x**2) - np.pi*np.cos(x)
    print('!!!!!!!!!! SanaaslamNoorInayatNoor(x^9 - pi*np.sin(x) - np.log(2.0+x^2) - 14.0), x =', SanaaslamNoorInayatNoor(f_SanaaslamNoorInayatNoor, df_SanaaslamNoorInayatNoor, 1.38))
    # x ≈ 1.08435594653098...
    # f_SanaaslamNoorInayatNoor = lambda x: x**7-np.exp(x)*np.cos(x)+np.log(x**2)-x/2
    # df_SanaaslamNoorInayatNoor = lambda x: 7*x**6 + 2/x + np.exp(x)*np.sin(x) - exp(x)*np.cos(x) - 1/2
    # print('SanaaslamNoorInayatNoor(x^7-exp(x)cos(x)+log(x^2)-x/2), x =', SanaaslamNoorInayatNoor(f_SanaaslamNoorInayatNoor, df_SanaaslamNoorInayatNoor, 1.1))

    print("------------------------------------------------")

    # x ≈ 1.08435594653098...
    f_abdulHassan = lambda x: x**7-np.exp(x)*np.cos(x)+np.log(x**2)-x/2
    df_abdulHassan = lambda x: 7*x**6 + 2/x + np.exp(x)*np.sin(x) - exp(x)*np.cos(x) - 1/2
    d2f_abdulHassan = lambda x: 42*x**5 - 2/x**2 + 2*np.exp(x)*np.sin(x)
    print('abdulHassan(x^7-np.exp(x)*np.cos(x)+log(x^2)-x/2), x =', abdulHassan(f_abdulHassan, df_abdulHassan, d2f_abdulHassan,1.0))

    print("------------------------------------------------")

    # x ≈ 1.34290728831020191244305715...
    f_HouLi = lambda x: x**7-np.sqrt(5.0-x)*x-x*np.sin(x)-4.0
    df_HouLi = lambda x: 7.0*x**6 - x/(2.0*np.sqrt(5.0-x)) - np.sqrt(5.0-x) - np.sin(x) - x*np.cos(x)
    print('HouLi(x^7-sqrt(5.0-x)*x-x*sin(x)-4.0), x =', HouLi(f_HouLi, df_HouLi, 1.3))

    # print("------------------------------------------------")
    # x ≈ 1.06806481313790...
    f_ghanbari = lambda x: x*np.exp(x) + np.sin(x)-x**3-3.0
    df_ghanbari = lambda x: -3.0*x**2+np.exp(x)*(x+1) + np.cos(x)
    # print('Ghanbari(x^3 + x*np.exp(x)+np.sin(x)+np.log(x+1)-1), x =', Ghanbari(f_ghanbari, df_ghanbari, 1.0))
    print("------------------------------------------------")

    # x ≈ 1.06806481313790...s
    f_alHusaynialSubaihiYSM4 = lambda x: x**3 + x*np.exp(x) + np.sin(x) + np.log(x+1.0)-1.0
    print('alHusaynialSubaihiYSM4(x^3 + x*np.exp(x)+np.sin(x)+np.log(x+1)-1), x =', alHusaynialSubaihiYSM4(f_alHusaynialSubaihiYSM4, 1.0))

    print("------------------------------------------------")

    # x ≈ 0.445675809154350159699962404...
    f_HeneritaPanday = lambda x: x**3 + x*np.exp(x)+(np.sin(x))/2.0-1.0
    df_HeneritaPanday = lambda x: 3.0*x**2 + np.exp(x)*(x+1)+0.5*np.cos(x)
    print('HeneritaPanday(x^7 + x*np.exp(x)+(sin(x))/2.0+np.log(x+1)-1.0), x =', HeneritaPanday(f_HeneritaPanday, df_HeneritaPanday, 0.34))

    print("------------------------------------------------")

    # x≈1.0814889981469337981
    f_MarojuBehlLMotsa = lambda x: x**8+np.sin(x)/pi - np.cos(x)-4.0
    df_MarojuBehlLMotsa = lambda x: 8.0*x**7 + np.sin(x) + np.cos(x)/pi
    print("MarojuBehlLMotsa(x^8+np.sin(x)/pi - np.cos(x)-4.0), root = ", MarojuBehlLMotsa(f_MarojuBehlLMotsa, df_MarojuBehlLMotsa, 1.0))

    print("------------------------------------------------")

    #x ≈ 0.8718077092312371...
    f_Thukralm = lambda x: 3*x**5-x*np.sin(x)**2-1
    df_Thukralm = lambda x: 15*x**4-sin(x)**2-2*x*np.sin(x)*np.cos(x)
    d2f_Thukralm = lambda x: 60*x**3+2*x*np.sin(x)**2-2*x*np.cos(x)**2-4*np.sin(x)*np.cos(x)
    d3f_Thukralm = lambda x: 180*x**2+6*np.sin(x)**2-6*np.cos(x)**2+8*x*np.sin(x)*np.cos(x)
    print('Thukralm(3x^5-x*np.sin(x)^2-1), x =', Thukralm(f_Thukralm, df_Thukralm, d2f_Thukralm, d3f_Thukralm, 0.85, 5))

    #x≈-5.5855867158953873777
    # f_Thukralm = lambda x: (x-3.0)**4+x**5-x*np.sin(x)**2+1.0
    # df_Thukralm = lambda x: 5.0*x**4 + 4.0*(x-3.0)**3-sin(x)**2-2.0*x*np.sin(x)*np.cos(x)
    # d2f_Thukralm = lambda x: 20.0*x**3+12.0*(x-3.0)**2+2.0*x*np.sin(x)**2-2.0*x*np.cos(x)**2-4.0*np.sin(x)*np.cos(x)
    # d3f_Thukralm = lambda x: 6.0*(10.0*x**2+4.0*x+np.sin(x)**2-12.0)-6.0*np.cos(x)**2+8.0*x*np.sin(x)*np.cos(x)

    # print('Thukralm((x-3)^4+x^5-x*np.sin(x)^2+1), x =', Thukralm(f_Thukralm, df_Thukralm, d2f_Thukralm, d3f_Thukralm,-5.5855))

    print("------------------------------------------------")

    # x≈1.0699028652198051727
    f_ZhouChenSong4 = lambda x: (x**3+np.exp(x)-2.0*x-2.0)**3
    df_ZhouChenSong4 = lambda x: 3.0*(3.0*x**2+np.exp(x)-2.0*x) * (x**3+np.exp(x)-2.0*x-2.0)**2
    print('ZhouChenSong4((x^3+np.exp(x)−2x-2)^3), x =', ZhouChenSong4(f_ZhouChenSong4, df_ZhouChenSong4, 1.0, 3))
    # 0.2575302854398607604553673049
    f_ZhouChenSong4 = lambda x: (x**2-np.exp(x)-3.0*x+2.0)**3
    df_ZhouChenSong4 = lambda x: -3.0*(-2.0*x+np.exp(x)+3.0)*(x**2-3.0*x-np.exp(x)+2.0)**2
    print('ZhouChenSong4(), x =', ZhouChenSong4(f_ZhouChenSong4, df_ZhouChenSong4, 0.25, 5))

    print("------------------------------------------------")

    # -4.8852118 à changer par x - sin(2x)^2 + 5x-sqrt(3)=0
    f_FerraraSharifiSalimi = lambda x: (x - np.power(sin(2.0*x),2) + 5.0)**3
    df_FerraraSharifiSalimi = lambda x: 3.0*(x-np.power(sin(2.0*x),2) + 5.0)**2 * (1.0 - 2.0*np.sin(4.0*x))
    print('FerraraSharifiSalimi(((x - np.sin(2x)^2 + 5.0)^3), x =', FerraraSharifiSalimi(f_FerraraSharifiSalimi, df_FerraraSharifiSalimi, -4.5, 3))
    
    print("------------------------------------------------")

    #x ≈ -1.12497018501971...
    f_hbm1 = lambda x: np.absolute(3.0*x**5-15.0*x*np.log(x**2)+np.cos(x)+1.0)
    print('HBM1(|3*x^5-15*x*np.log(x^2)+np.cos(x)+1)|), root:\n', HBM1(f_hbm1, -0.5, 1.5))

    # f_hbm2 = lambda x: np.absolute(3.0*x**5-15.0*x*np.log(x**2)+np.cos(x)+1.0)
    # print('HBM2(|3*x^5-15*x*np.log(x^2)+np.cos(x)+1)|), root: {.10f}\n', HBM2(f_hbm2, -1.2, 0.0075, 20, 1e-8))
    exit()
    # ------------------------------------------------------------------------
    # z**10-4*z**9+10*z**8-5*z**7+11*z**6-7*z**5+13*z**4-2z**3+27*z**2-14*z+2=0
    # z≈-0.89887 - 0.68379 i
    # z≈-0.89887 + 0.68379 i
    # z≈-0.22568 - 1.22247 i
    # z≈-0.22568 + 1.22247 i
    # z≈0.25345 - 0.08233 i
    # z≈0.25345 + 0.08233 i
    # z≈0.92395 - 0.90094 i
    # z≈0.92395 + 0.90094 i
    # z≈1.9471 - 2.1881 i
    # z≈1.9471 + 2.1881 i
    # coefficients = np.array([1, -17, -12, 962, 461, -16281, -38466, -22680])
    # initials = np.array([-0.6-0.3*1j, -0.6+0.3*1j, -0.2-1.0*1j, -0.2+1.0*1j, 0.4-0.1*1j, 0.4+0.1*1j, 0.5-0.7*1j, 0.5+0.7*1j, 1.5-2.4*1j, 1.5+2.4*1j], dtype=np.complex128)
    # ------------------------------------------------------------------------
    np.set_printoptions(precision=2, suppress = True)

    print("------------------------------------------------")
    print(">>>> DURaND-KERNER")
    # (x+1)(x+2)(x+3)(x+5)(x-7)(x-9)(x-12)=0
    # x^7 - 17x^6 - 12x^5 + 962x^4 + 461x^3 - 16281x^2 - 38466x - 22680
    # coefficients = np.array([1, -17, -12, 962, 461, -16281, -38466, -22680])
    # initials = np.array([-0.6+0j, -1.4+0j, -2.5+0j, -4.6+0j, 6.3+0j, 8.3+0j, 12.9+0j], dtype=np.complex128)

    # x^6 + 2x^4 + 6x^3 - 9x^2 + 3x - 1 = 0
    # coefficients = np.array([1, 2, 6, -9, 3, -1])
    # initials = np.array([-1.7+0j, 0.6+0j, 0.1-0.3j, 0.1+0.3j, 0.3-1.9j, 0.3+1.9j], dtype=np.complex128)
    # x ≈ -1.96332
    # x ≈ 0.830442
    # x ≈ 0.14323 - 0.33867 i
    # x ≈ 0.14323 + 0.33867 i
    # x ≈ 0.4232 - 2.0874 i
    # x ≈ 0.4232 - 2.0874 i

    # (x-1)*(x+(2+3i))(x+(2-3i))*(x+i)*(x-i)*(x^2-4x+3)*(x-2-i)*(x-2+i)=0
    # x^9 - 5x^8 + 10x^7 - 50x^6 + 248x^5 - 600x^4 + 790x^3 - 750x^2 + 551x - 195 = 0
    coefficients = np.array([1, -5, 10, -50, 248, -600, 790, -750, 551, -195])
    # initials = cerclecomplex(9)
    initials = np.array([-0.8, 4.6, -1.7-2.8j, -1.7+2.8j, -0.9j, 0.9j, 1.6-0.8j, 1.6+0.8j, ], dtype=np.complex128)
    # initials = aberth_initial(coefficients)
    # x = 1
    # x = 3
    # x = -2-3i
    # x = -2+3i
    # x = -i
    # x = i
    # x = 2-i
    # x = 2+i

    # x^12+9x^11-3x^10+4x^9+6x^8-2x^7+5x^6-7x^5+2x^4-8x^3+4x^2-5x-1=0
    # coefficients = np.array([1, 9, -3, 4, 6, -2, 5, -7, 2, -8, 4, -5, -1])
    # x = 1
    # x ≈ -9.35795
    # x ≈ -1.2378
    # x ≈ -0.168915
    # x ≈ -0.61265 - 0.63398 i
    # x ≈ -0.61265 + 0.63398 i
    # x ≈ -0.02646 - 0.95454 i
    # x ≈ -0.02646 + 0.95454 i
    # x ≈ 0.41540 - 0.88022 i
    # x ≈ 0.41540 + 0.88022 i
    # x ≈ 0.60603 - 0.62762 i
    # x ≈ 0.60603 + 0.62762 i

    WeierstrassDurandKerner(coefficients)
    k = 20
    tol: np.float64 = np.finfo(np.float64).resolution
    # print("------------------------------------------------")
    # print(">>>> aBERTH-EHRLICH")
    # aberthEhrlich(coefficients, initials, k, tol)
    print("------------------------------------------------")
    print(">>>> NOUREIN")
    coefficients = np.array([np.complex128(1+0j), np.complex128(4+2j), -np.complex128(11-8j), -np.complex128(64-12j), -np.complex128(37-8j), -np.complex128(68-130j), -np.complex128(1105+0j)], dtype=np.complex128)
    # Nourein(coefficients)
    print("------------------------------------------------")
    print(">>>> NM10D")
    NM10D(coefficients, initials, k, tol)

    print("------------------------------------------------")

    # a, b = new_nosymmetric_linearsystem(16)
    # print('a :\n', a)
    # print('b :\n', b)

    print("------------------------------------------------")
    m = np.array([[1,2,3],[4,5,6],[7,8,9]])
    a, b = systeme_lineaire_non_symetrique(3)
    print(f"{a=}")
    print(f"{b=}")
    x = np.linalg.solve(a,b)
    print(f"{x=}")

    print("------------------------------------------------")

    #f_mnm = lambda x: ((x-1)**3-sin(x)*np.log(x))**4 # x = 1 (m=3)
    #print('mnm(f,100,5,1e-8), x =', mnm(f_mnm,2,100,1e-8), '\n')

    # print("------------------------------------------------")

    print("------------------------------------------------")
    a = np.array([[1,-1,0],[-1,1,-1],[0,-1,1]])
    # a = np.array([[2 ,2 ,1] ,[2 ,1 ,4] ,[1 ,4 ,2]])
    # a = np.array([[2,5,0,-1],[-3,5,11,9],[8,5,-2,1],[7,-6,4,-8]])
    q, r = qr_gram_schmidt_classique(a)
    print("CGS :\n-----")
    print("a = {}\n".format(np.array_str(a, precision=6)))
    print("Q = {}\n".format(np.array_str(q, precision=6)))
    print("R = {}\n".format(np.array_str(r, precision=6)))
    print("QR = {}".format(np.array_str(q@r, precision=6)))

    # a = np.array([[1,-1,0],[-1,1,-1],[0,-1,1]])
    # q, r = qr_gram_schmidt_modified(a)
    # #q.shape, r.shape
    # print('MGS :')
    # print('q=',np.around(q,decimals=8),'\n')
    # print('r=',np.around(r,decimals=8),'\n')
    # print("Q : ", est_ortho(q))

    # a = np.array([[2,5,0,-1],[-3,5,11,9],[8,5,-2,1],[7,-6,4,-8]])
    q, r = qr_gram_schmidt_modifiee(a)
    #q.shape, r.shape
    print("MGS :\n-----")
    print("a = {}\n".format(np.array_str(a, precision=6)))
    print("Q = {}\n".format(np.array_str(q, precision=6)))
    print("R = {}\n".format(np.array_str(r, precision=6)))
    print("QR = {}".format(np.array_str(q@r, precision=6)))

    print("------------------------------------------------")

    a = np.array([[3,8,-1],[2,9,0],[7,0,0]])
    b = np.array([[6,10,-3],[-5,-1,0],[8,0,0]])

    print("------------------------------------------------")

    # a = np.array([[2,-1,3],[2,-5,-3],[1,3,2]])
    a = np.array([[2,5,0,-1],[-3,5,11,9],[8,5,-2,1],[7,-6,4,-8]])
    Q,R = qr_householder(a)
    print('Q Householder :\n', np.around(Q, decimals=8))
    print('R Householder :\n', np.around(R, decimals=8))
    print('QR Householder :\n', Q@R)

    print("------------------------------------------------")

    a = np.array([[2,5,0,-1],[-3,5,11,9],[8,5,-2,1],[7,-6,4,-8]])
    Q,R = qr_givens(a)
    print('Q Givens :\n', np.around(Q, decimals=8))
    print('R Givens :\n', np.around(R, decimals=8))
    print('QR Givens :\n', Q@R)

    print("------------------------------------------------")

    x = np.array([30,-4,8,-31,24,-13])
    print(householder_reflecteur(x))

    print("------------------------------------------------")

    a = np.array([[1,2,-1],[2,0,2],[-1,2,1]])
    print("Polynôme caractéristique de a :")
    print(faddeev_leverier(a))

    print("------------------------------------------------")

    a = np.array([[1,-2,-1],[-2,0,2],[-1,2,1]])
    # a = np.array(
    #     [[-11.22497216, 0.62360956, 1.87082869, 6.85970521],
    #     [ 0., -10.51718171, -1.88580305, -8.43593129],
    #     [ 0., 0., 11.57340688, 3.13288577],
    #     [ 0., 0., 0., 4.35482888]])
    # a = np.array([[3.556,-1.778,0],[-1.778,3.556,-1.778],[0,-1.778,3.556]])
    # 4/-1,1,1
    # -2/-1,-2,1
    # 0/1,0,1
    # a = np.array([[2,2,0],[-1,-4,1],[3,0,-1]])
    # -3/-2,5,3
    # -2/-1,2,3
    # 2/1,0,1
    s, v = power(a,1e-16,20)
    print("Plus grande valeur propre : %0.8f\t" % s)
    print("et son vecteur propre :")
    for i in range(len(v)):
        print("%0.8f\t" % v[i])

    s, v = power(a,1e-8,20)
    w,v=np.linalg.eig(a)
    print('E-value:', w)
    print('E-vector', v)

    print("------------------------------------------------")

    a = np.array([[1,-2,-1],[-2,0,2],[-1,2,1]])
    # 4/-1,1,1
    # -2/-1,-2,1
    # 0/1,0,1
    # a = np.array([[2,2,0],[-1,-4,1],[3,0,-1]])
    # -3/-2,5,3
    # -2/-1,2,3
    # 2/1,0,1

    #print("Plus petite valeur propre et vecteur propre :", invpower(a,1e-8,20))
    print("------------------------------------------------")

    m = np.array([[1,0,2],[-2,1,1],[-1,-1,2]])
    v = np.array([-2,4,7])
    print("matrice * vecteur = \n", prodvecmat(m,v))

    print("------------------------------------------------")

    a = np.array([[3,8,-1],[2,9,0],[7,0,0]])
    b = np.array([[6,10,-3],[-5,-1,0],[8,0,0]])
    print('a=',a)
    print('b=',b)
    print('mattriupproduct(ab) = ', mattriupproduct(a,b), '\n')

    print("------------------------------------------------")

    # a = np.random.randint(1,5,[4,4])
    # b = np.random.randint(1,5,[4,4])
    # print('a=',a)
    # print('b=',b)
    # print('ab = ', matproduct(a,b), '\n')

    print("------------------------------------------------")

    a = np.random.randint(-10,10,[8,8])
    b = np.random.randint(-10,10,[8,8])
    print('a=',a)
    print('b=',b)
    print('ab = ', blockmatproduct(a,b,2), '\n')
    print('ab = ', blockmatproduct(a,b,4), '\n')

    print("------------------------------------------------")

    a = np.array([[1,2],[3,4]])
    b = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    print('Kronecker(a,b) = ', kroneckerproduit(a,b))
    print('Kronecker2(a,b) = ', np.kron(a,b))

    print("------------------------------------------------")

    # a = np.random.uniform(low=-100, high=100, size=(10, 10))
    # b = np.random.uniform(low=-100, high=100, size=(10, 10))
    a = np.random.randint(-10,10,[8,8])
    b = np.random.randint(-10,10,[8,8])
    # print("strassen(a,b) = ", strassen(a,b))

    print("------------------------------------------------")

    matriinf = np.array([[1,0,0,0], [-10,1,0,0], [16,-12,-1,0], [-2,6,2,1]])

    print(f"{inv_tri_inf(matriinf)}") #OK

    print("------------------------------------------------")
    matrisup = np.array([[1, -4, 8, 1], [0, 1, -5, -2], [0, 0, -1, 1], [0, 0, 0, -1]])

    print(f"{inv_tri_sup(matrisup)}") #OK

    print("------------------------------------------------")

    a = np.array(
                [[-2, -7, -5, -4],
                [-6, -6,  9, 3],
                [ 0,  6,  1, 8],
                [ 9,  7, -2, 9]])

    print('Det(a) = ', bareiss(a))

    print("------------------------------------------------")

    # a = np.array([[1,2,3],[4,5,6],[7,8,9]])
    # print("Det(a) :", Dodgson(a))

    # print("------------------------------------------------")

    a = np.array([[3,0,0],[-10,5,0],[7,-6,3]])
    b = np.array([12,-15,16])
    print(a)
    print(b)
    print('Substitutionavant(a,b,x) =', Substitutionavant(a,b)) #4 5 6

    print("------------------------------------------------")
    # a = np.array([[4,-1,2],[0,8,5],[0,0,9]])
    # b = np.array([8,31,27])

    a = np.array([[3, 2, 1], [0, 5, 4], [0, 0, 2]])

    b = np.array([10, 49, 12])

    print(a)
    print(b)
    print('Substitutionarriere(a,b,x) =', Substitutionarriere(a,b)) # 1 2 3

    print("------------------------------------------------")

    a = np.array([[3,-2,-1,4],
                [0,2,1,-1],
                [0,0,-2,3],
                [0,0,0,1]])
    print('inv tri sup(a) =\n', inv_tri_sup(a))

    print("------------------------------------------------")

    a = np.array([[1,0,0,0],
                [-2,3,0,0],
                [2,1,-1,0],
                [3,-2,-1,4]])
    print('inv tri inf(a) = ', inv_tri_inf(a))

    print("------------------------------------------------")
    a = np.array([[3, 1, -2, 0],
                    [2, -1, 2, 0],
                    [-2, 1, -3, 1],
                    [0, 0, -1, 5]])

    inva = GaussJordan(a)
    print("Pivot(x) = ", GaussJordan(a))

    print("------------------------------------------------")

    # a = np.array([[-3, 0, -8, 0],
    #             [-6, 1, -6, 6],
    #             [5, -6, -5, -2],
    #             [4, 0, -4, 3]])
    # #x = [4, 0, 0, 1]
    # b = np.array([-12, -18, 18, 19])
    a = np.array([[3, -1, 9],
                [7, 2, -4],
                [-5, 8, 6]])
    #x = [1, 4, 7]
    b = np.array([62, -13, 69])
    print("Pivot(x) =", pivotgauss(a,b))

    print("------------------------------------------------")

    a = np.array([[2, 1, -8, -1],
    [-3, 1, -6, 2],
    [ 5, -2, -5, -2],
    [-8, -2, 5, 6]])
    # x = [ 5 -3  6  4]
    b = np.array([-45, -46, -7, 20])
    print("LU(x) =", lu(a,b))

    print("------------------------------------------------")

    a = np.array([[9,3,12],[3,50,39],[12,39,122]])
    b = np.array([87, 351, 751])
    # a = np.array([[3,1,-1,2],[1,7,-2,-6],[-1,-2,9,5],[2,-6,5,8]])
    # b = np.array([10,-15,42,37]) x = 1,2,3,4
    # a = np.array(
    # [[ 31, -9,  4, -4,  4, -1, -4, -3, -3, -5 ],
    # [ -9, 28,  3,  2, -5,  4, -4, -6,  8,  0 ],
    # [  4,  3, 24, -9,  6, -1, -8, -6,  7, -2 ],
    # [ -4,  2, -9, 34,  3, -9, -1, -7,-10, -9 ],
    # [  4, -5,  6,  3, 38,  0, -2,  1,  8, -7 ],
    # [ -1,  4, -1, -9,  0, 39,  7,  1,  8, -1 ],
    # [ -4, -4, -8, -1, -2,  7, 20, -5, -8, -6 ],
    # [ -3, -6, -6, -7,  1,  1, -5, 21,  1,  3 ],
    # [ -3,  8,  7, -10,  8,  8, -8, 1, 26, -4 ],
    # [ -5,  0, -2, -9, -7, -1, -6,  3, -4, 29 ]]
    # )
    # b = np.array([ 135, -84, 108, -48, -134, 310, 63, -104, 155, -268])
    # x = [ 2 -9  5  3 -9  8  0 -4  8 -8]
    a,b = new_symmetricpositivedefinite_linearsystem(4)
    print("Cholesky(x) =", cholesky(a,b))

    print("------------------------------------------------")

    # a = np.array([[2,-5,1,5],[-3,3,-4,1],[8,-7,-1,1],[1,5,2,9]])
    # b = np.array([65,-10,45,70]) # 2, -3, 1, 9
    # print('a=', a)
    # print('b=', b)
    # print('Bareiss = ', gauss_bareiss(a,b), '\n')

    # print("------------------------------------------------")

    # 3   1   0   0   0
    # 1  -1   4   0   0
    # 0   4   2  -2   0
    # 0   0  -2   5   3
    # 0   0   0   3   4

    # symétrique
    c = np.array([0,1,4,-2,3])
    d = np.array([3,-1,2,5,4])
    e = np.array([1,4,-2,3,0])
    b = np.array([-20,0,10,-18,-15])

    # for i in range(n):
    #     a[i,i] = -2 # Diagonal
    #     if i<n:
    #         a[i,i+1] = 1 # sous-diagonale
    #     if i>1:
    #         a[i,i-1] = 1 # sur-diagonale

    print("Thomas(x) =", thomas(c,d,e,b)) # -7, 1, 2, -1, -3
    
    # print("------------------------------------------------")

    # e = np.array([0,0,3,6,3,2,-8])
    # f = np.array([0,2,2,1,-7,1,3])
    # d = np.array([2,1,-2,2,1,3,4])
    # g = np.array([-1,3,1,-3,-3,5,0])
    # h = np.array([4,1,4,1,4,0,0])
    # b = np.array([-1,6,25,2,-2,30,10])
    #print('Pentad(x) =', pentad(e,f,d,g,h,b)) # x = 2 1 -1 4 3 -2 5

    print("------------------------------------------------")
    dim = 10
    a, b = systeme_lineaire_non_symetrique(dim)
    chrono = time.time(); x = gauss_seydel(a,b); print(f"Durée GS(x) > ", time.time()-chrono)
    chrono = time.time(); x = sor(a,b,1.9,dim); print(f"Durée SOR(x) > ", time.time()-chrono)
    chrono = time.time(); x = aor(a,b,0.7,1.3,dim); print(f"Durée aOR(x) > ", time.time()-chrono)

    print("------------------------------------------------")

    n = 10
    a = np.array(
    [[ 31, -9,  4, -4,  4, -1, -4, -3, -3, -5 ],
    [ -9, 28,  3,  2, -5,  4, -4, -6,  8,  0 ],
    [  4,  3, 24, -9,  6, -1, -8, -6,  7, -2 ],
    [ -4,  2, -9, 34,  3, -9, -1, -7,-10, -9 ],
    [  4, -5,  6,  3, 38,  0, -2,  1,  8, -7 ],
    [ -1,  4, -1, -9,  0, 39,  7,  1,  8, -1 ],
    [ -4, -4, -8, -1, -2,  7, 20, -5, -8, -6 ],
    [ -3, -6, -6, -7,  1,  1, -5, 21,  1,  3 ],
    [ -3,  8,  7, -10,  8,  8, -8, 1, 26, -4 ],
    [ -5,  0, -2, -9, -7, -1, -6,  3, -4, 29 ]], dtype=np.float64
    )
    b = np.array([ 135, -84, 108, -48, -134, 310, 63, -104, 155, -268], dtype=np.float64)
    # x = [ 2 -9  5  3 -9  8  0 -4  8 -8]

    # a = scipy.io.mmread("/Users/philippepeter/Downloads/nasa2146/matrix.mtx").astype('float32')
    # b = scipy.io.mmread("/Users/philippepeter/Downloads/nasa2146/nasa2146_b.mtx").astype('float32')
    # print(a.a)
    # if isinstance(a, scipy.sparse.coo_matrix):
    #     print("Matrix is in COO format")
    # else:
    #     print("Matrix is not in COO format")
    # print(f"{a.shape[0]=}")

    # a_mm, b_mm = LireMatrixMarket("/Users/philippepeter/Desktop/Pres_Poisson/")
    # a = np.array(a_mm.toarray())
    # b = np.array(b_mm.toarray())
    # print(np.info(a_mm))
    # print(np.info(b_mm))
    # print("Cholesky2(x) =", cholesky(a,b))

    # n = 100
    # a, b = new_symmetricpositivedefinite_linearsystem(n)

    # print("GradientConjugue(x)")
    # chrono = time.time(); x = GradientConjugue(a, b); print("Durée :", time.time()-chrono)
    # print(x)
    # print("------------------------------------------------")
    # print("GradientConjuguePreconditionneJacobi(x)")
    # chrono = time.time(); x = GradientConjuguePreconditionneJacobi(a, b); print("Durée :", time.time()-chrono)
    # print(x)
    # print("------------------------------------------------")
    # print("GradientConjuguePreconditionneSSOR(x)")
    # chrono = time.time(); x = GradientConjuguePreconditionneSSOR(a, b, 1.2); print("Durée :", time.time()-chrono)
    # print(x)
    # print("------------------------------------------------")
    # print("GradientConjuguePreconditionneICholesky1(x)")
    # chrono = time.time(); x = GradientConjuguePreconditionneIC1(a, b); print("Durée :", time.time()-chrono)
    # print(x)
    # print("------------------------------------------------")
    # print("GradientConjuguePreconditionneICholesky2(x)")
    # chrono = time.time(); x = GradientConjuguePreconditionneIC2(a, b); print("Durée :", time.time()-chrono)
    # print(x)

    print("------------------------------------------------")
 
    a = np.array([[3,-3,1,5,2,2],[1,2,-2,9,2,1],[6,9,3,8,1,3],[-2,3,3,2,-4,4],[5,-6,-2,9,8,9],[4,-5,-3,7,-4,-1]])
    b = np.array([30,-4,8,-31,24,-13])
    print('BICR(x) =', bicr(a,b,100)) # x = 2, -3, 8, 1, 6, -5

    print("------------------------------------------------")
    a = np.array([[3,-3,1,5,2,2],[1,2,-2,9,2,1],[6,9,3,8,1,3],[-2,3,3,2,-4,4],[5,-6,-2,9,8,9],[4,-5,-3,7,-4,-1]])
    b = np.array([30,-4,8,-31,24,-13])
    print('CGS(x) =', cgs(a,b,100)) # x = 2, -3, 8, 1, 6, -5

    print("------------------------------------------------")

    a = np.array([[3,-3,1,5,2,2],[1,2,-2,9,2,1],[6,9,3,8,1,3],[-2,3,3,2,-4,4],[5,-6,-2,9,8,9],[4,-5,-3,7,-4,-1]])
    b = np.array([30,-4,8,-31,24,-13])
    print('CGS2(x) =', cgs2(a,b)) # x = 2, -3, 8, 1, 6, -5

    # print("------------------------------------------------")

    a = np.array([[3,-3,1,5,2,2],[1,2,-2,9,2,1],[6,9,3,8,1,3],[-2,3,3,2,-4,4],[5,-6,-2,9,8,9],[4,-5,-3,7,-4,-1]])
    b = np.array([30,-4,8,-31,24,-13])
    print('BICGSTaB(x) =', bicgstab(a,b,100)) # x = 2, -3, 8, 1, 6, -5

    print("------------------------------------------------")

    a = np.array([[3,-3,1,5,2,2],[1,2,-2,9,2,1],[6,9,3,8,1,3],[-2,3,3,2,-4,4],[5,-6,-2,9,8,9],[4,-5,-3,7,-4,-1]])
    b = np.array([30,-4,8,-31,24,-13])
    print('QMRCGSTaB(x) =', qmrcgstab(a,b,100)) # x = 2, -3, 8, 1, 6, -5

    print("------------------------------------------------")

    a = np.array([[3,-3,1,5,2,2],[1,2,-2,9,2,1],[6,9,3,8,1,3],[-2,3,3,2,-4,4],[5,-6,-2,9,8,9],[4,-5,-3,7,-4,-1]])
    b = np.array([30,-4,8,-31,24,-13])
    print('GPBICG(x) =', gpbicg(a,b,100)) # x = 2, -3, 8, 1, 6, -5

    print("------------------------------------------------")

    a = np.array([[3,-3,1,5,2,2],[1,2,-2,9,2,1],[6,9,3,8,1,3],[-2,3,3,2,-4,4],[5,-6,-2,9,8,9],[4,-5,-3,7,-4,-1]])
    b = np.array([30,-4,8,-31,24,-13])
    print('CRS(x) =', crs(a,b,100)) # x = 2, -3, 8, 1, 6, -5

    print("------------------------------------------------")

    a = np.array([[3,-3,1,5,2,2],[1,2,-2,9,2,1],[6,9,3,8,1,3],[-2,3,3,2,-4,4],[5,-6,-2,9,8,9],[4,-5,-3,7,-4,-1]])
    b = np.array([30,-4,8,-31,24,-13])
    print('CORS(x) =', cors(a,b,100)) # x = 2, -3, 8, 1, 6, -5

    print("------------------------------------------------")

    # a = np.array([[3,-3,1,5,2,2],[1,2,-2,9,2,1],[6,9,3,8,1,3],[-2,3,3,2,-4,4],[5,-6,-2,9,8,9],[4,-5,-3,7,-4,-1]])
    # b = np.array([30,-4,8,-31,24,-13])
    # print('BICGSTaB2(x) =', bicgstab2(a,b)) # x = 2, -3, 8, 1, 6, -5

    # print("------------------------------------------------")

    a = np.array([[3,-3,1,5,2,2],[1,2,-2,9,2,1],[6,9,3,8,1,3],[-2,3,3,2,-4,4],[5,-6,-2,9,8,9],[4,-5,-3,7,-4,-1]])
    b = np.array([30,-4,8,-31,24,-13])
    # print('BICGSTaBL(x) =', bicgstabl(a,b,4)) # x = 2, -3, 8, 1, 6, -5

    # print("------------------------------------------------")
    # a = np.array([[3,-3,1,5,2,2],[1,2,-2,9,2,1],[6,9,3,8,1,3],[-2,3,3,2,-4,4],[5,-6,-2,9,8,9],[4,-5,-3,7,-4,-1]])
    # b = np.array([30,-4,8,-31,24,-13])
    # print('ORTHOMIN(x) =', orthomin(a,b)) # x = 2, -3, 8, 1, 6, -5

    # print("------------------------------------------------")
    # a = np.array([[3,-3,1,5,2,2],[1,2,-2,9,2,1],[6,9,3,8,1,3],[-2,3,3,2,-4,4],[5,-6,-2,9,8,9],[4,-5,-3,7,-4,-1]])
    # b = np.array([30,-4,8,-31,24,-13])
    # print('TFQMR(x) =', tfqmr(a,b)) # x = 2, -3, 8, 1, 6, -5

    # print("------------------------------------------------")
    a = np.array([[3,-3,1,5,2,2],[1,2,-2,9,2,1],[6,9,3,8,1,3],[-2,3,3,2,-4,4],[5,-6,-2,9,8,9],[4,-5,-3,7,-4,-1]])
    b = np.array([30,-4,8,-31,24,-13])
    print('TFQMORS(x) =', tfqmors(a,b)) # x = 2, -3, 8, 1, 6, -5

    print("------------------------------------------------")

    # x = np.array([0.5, 1.1, 1.3, 1.6, 1.9, 2.2, 2.5, 2.6, 2.9, 3.2, \
    #                 3.3, 3.6, 3.9, 4.1, 4.7, 4.8, 5.1, 5.5, 5.8, 6.1])
    # y = np.array([0.7, 0.8, 1.4, 1.9, 1.8, 2.5, 2.4, 2.7, 3.3, 3.4, \
    #                 4.2, 3.7, 4.2, 4.8, 5.1, 5.6, 5.8, 5.9, 6.2, 7.2])
    # x = np.array([0.05, 0.18, 0.31, 0.42, 0.5])
    # y = np.array([0.12, 0.22, 0.35, 0.38, 0.49])
    x = np.array([1, 3, 4, 5, 7])
    y = np.array([2, 3, 4, 6, 8])
    a,b = reglin(x,y)
    print("REGLIN : y = %.4f"%a, "x + %.4f"%b)

    print("------------------------------------------------")

    x = np.array([1.9, 2.1, 2.7, 2.9, 3.9, 4.9, 6.8, 7.8, 9.1, 9.2, 10.2, 10.2])
    y = np.array([11.1, 9.4, 7.6, 5.9, 3.8, 1.9, 1.6, 3.3, 5.3, 7.1, 8.8, 10.9])
    a,b,c = regqua(x,y)
    print("y = %.4f" % a, "x^2 + %.4f" % b, "x + %.4f" % c) # y = 0.5174x2 - 6.3166x + 20.6094 : r = 0.960864

    print("------------------------------------------------")

    a = np.array([[1,1,1,1],[1,2,2,2],[1,2,3,3],[1,2,3,4]])
    print('a=', a)
    print('V : ', jacobi(a,50)) # lambda=2+np.sqrt(3),2,2-np.sqrt(3)

    print("------------------------------------------------")

    # eigenvalues_givens_householder(a)lagrange(x, y, xx, 4)
    # rbezier_curve(degree)
    # cubicspline(points)

    # print("------------------------------------------------")

    x = np.array([1, 2, 3, 4])
    y = np.array([4.42, 6.26, 7.67, 8.85])
    t = 2.6
    print('Lagrange1D =', Lagrange1D(x, y, t))

    print("------------------------------------------------")
    dec = np.finfo(np.longdouble).precision

    # f_trap = lambda x: x**3.0
    # print("trapezes(x^3,-2,4,1000) =", trapezes(f_trap, -2, 4, 1000, dec))
    # f_trap = lambda x: x*x*np.exp(-x)*np.sin(x)
    # print('trapezes(x*x*np.exp(-x)*np.sin(x),1,3,20000) = ', trapezes(f_trap,1,3,200000, dec))
    #    f_trap = lambda x: np.exp(-x**2)
    #    print('trap = ', trapezes(f_trap,-1,1,100, dec), '\n')

    print("------------------------------------------------")

    f_simp13 = lambda x: x**2-np.sqrt(2*x)+np.cos(np.pi*x)-np.sin(4*np.pi*x) + 4
    print("simp13(x**5-x**3*np.cos(x**2),0,pi,1000) =", simpson13(f_simp13, 0, 1, 1000))

    print("------------------------------------------------")
    f_simp13_v2 = lambda x: x**2-np.sqrt(2*x)+np.cos(np.pi*x)-np.sin(4*np.pi*x) + 4
    print("simp13_v2(x**5-x**3*np.cos(x**2),0,1,1000) =", simpson13_v2(f_simp13_v2, 0, 1, 1000))

    print("------------------------------------------------")

    f_simp38 = lambda x: x**2-np.sqrt(2*x)+np.cos(np.pi*x)-np.sin(4*np.pi*x) + 4
    print("simp38(x**5-x**3*np.cos(x**2),0,1,1000) =", simpson38(f_simp38, 0, 1, 1000))

    print("------------------------------------------------")

    f_romb = lambda x: 4.0 / (1.0+x**2)
    print("romberg(f_romb,0,1,10,1e-8) = ", romberg(f_romb, 0.0, 1.0, 10, dec))

    print("------------------------------------------------")

    f_gl = lambda x: x**2-np.sqrt(2*x)+np.cos(np.pi*x)-np.sin(4*np.pi*x) + 4
    print("gauss_legendre(f_gl,0,1) =", gauss_legendre(f_gl,0,1))
    #    f_gl = lambda x: np.exp(-x**2)
    #    print('gleg = ', gauss_legendre(f_gl,-1,1), '\n')

    print('P(3,2) = ', legendre_eval(3,2))
    # sd = 50
    # pnt = [[0,0],[1,1],[1,0]]
    # t = np.array([i*(1/sd) for i in range(0,sd+1)])

    # bx = (1.0-t)*((1.0-t)*pnt[0][0]+t*pnt[1][0])+t*((1.0-t)*pnt[1][0]+t*pnt[2][0])
    # by = (1.0-t)*((1.0-t)*pnt[0][1]+t*pnt[1][1])+t*((1.0-t)*pnt[1][1]+t*pnt[2][1])
    # plot(bx, by)
    # pp.xlim(-.1,1.1)
    # pp.xlim(-.1,1.1)
    # pp.savefig('Bézier', dpi=100)
    # pp.show()

    # mc(samples)
    # qmc(samples,dim)
    # mcmetropolis()

    sys.exit(None)
