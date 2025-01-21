import numpy as np
import time as t

def GradientConjugue(
    a: np.ndarray,
    b: np.ndarray,
    tol:np.float64 = np.finfo(np.float64).eps
    ) -> np.ndarray:

    err1:str = "A n'est pas en 2D !"
    err2:str = "A est vide !"
    err3:str = "A est rectangulaire !"
    err4:str = "A n'est pas symétrique."
    err5:str = "A n'est pas SPD."
    err6:str = "Ordre de A != ordre de b."

    if a.ndim != 2:
        raise np.linalg.LinAlgError(err1)

    if a.size == 0:
        raise np.linalg.LinAlgError(err2)

    n, m = a.shape
    if n != m:
        raise np.linalg.LinAlgError(err3)

    if np.allclose(a, a.T) is False:
        raise np.linalg.LinAlgError(err4)

    if np.any(np.linalg.eigvals(a)) <= 0.0:
        raise np.linalg.LinAlgError(err5)

    if n != b.shape[0]:
        raise np.linalg.LinAlgError(err6)
    
    x = np.ones(n)
    r = b - a @ x
    d = np.copy(r)
    rho0 = np.dot(r,r)

    while True:
        ad = np.dot(a,d)
        add = np.dot(ad,d)
        alpha = rho0 / add
        x += alpha * d
        r -= alpha * ad
        rho1 = np.dot(r,r)
        if np.allclose(np.linalg.norm(r), np.zeros_like(b), atol=tol) == True:
            break
        else:
            beta = rho1 / rho0
            d = r + d * beta
            rho0 = rho1

    return x

def creer_systeme_lineaire_spd(n: int) -> np.ndarray:
    rng = np.random.default_rng()

    x: np.ndarray = rng.random(size=n)
    d: np.ndarray = np.diag(x)
    m: np.ndarray = rng.random(size=(n,n))
    tu: np.ndarray = np.triu(m, k = 1) / n + (d**2) * n
    a = (tu @ tu.T) + (d**2)
    b = a @ x

    return a.astype(np.float64), \
            b.astype(np.float64), \
            x.astype(np.float64)

if __name__ == "__main__":
    aff = False
    ordre = 500
    prec = np.finfo(np.float32).precision
    np.set_printoptions(precision=prec)

    a, b, x = creer_systeme_lineaire_spd(ordre)
    pa = np.array_str(a, precision=prec, suppress_small=True)
    px = np.array_str(x, precision=prec, suppress_small=True)
    pb = np.array_str(b, precision=prec, suppress_small=True)
	# print(f"A = {pa}\n")
	# print(f"b = {pb}\n")

    debut = t.perf_counter()
    x1 = GradientConjugue(a, b)
    fin = t.perf_counter()
    np.set_printoptions(precision=prec, suppress = True)
    temps = np.format_float_positional(np.float16(fin - debut))
    print(f"CG : {temps} s")
    print("Vérification : \nx = ", end="")
    if aff:
        print(f"{px}\n")
        print(f"{x1=}\n")
    ret = np.allclose(x1, x, np.finfo(np.float64).eps)
    if ret:
        print("OK")
    else:
        print("KO")
    # np.savetxt("/Users/philippepeter/Desktop/x_cgs.txt", x3, fmt="%f", delimiter=", ")

