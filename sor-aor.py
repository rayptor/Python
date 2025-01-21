import numpy as np
import time as t


def sor(
    a: np.ndarray,
    b: np.ndarray,
    omega: float,
    k: int = 100,
    tol: float = np.finfo(np.float64).eps
    ) -> np.ndarray:

    err1:str = "A n'est pas en 2D !"
    err2:str = "A est vide !"
    err3:str = "A est rectangulaire !"
    err4:str = "Ordre de A != ordre de b."
    err5:str = "A n'est pas à diagonale dominante !"

    if a.ndim != 2:
        raise np.linalg.LinAlgError(err1)

    if a.size == 0:
        raise np.linalg.LinAlgError(err2)

    n, m = a.shape
    if n != m:
        raise np.linalg.LinAlgError(err3)

    if n != b.shape[0]:
        raise np.linalg.LinAlgError(err4)

    for i in range(n):
        diag = np.fabs(a[i,i])
        horsdiag = np.sum(np.fabs(a[i,:])) - diag
        if not np.greater_equal(diag, horsdiag):
            raise np.linalg.LinAlgError(err5)

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
            frac = (omega * (b[i] - s)) / a[i,i]
            x1[i] = (1 - omega) * x0[i] + frac

        norme = np.linalg.norm(b - a @ x1)
        if np.less(norme, tol) == True or it == k:
            break
        else:
            x0 = x1
            it += 1

    return x1.astype(np.float64)

# https://www.joezhouman.com/2021/11/21/NumericalAnalysisIteration.html
def aor(
    a: np.ndarray,
    b: np.ndarray,
    omega: float,
    r: float,
    k: int = 100,
    tol: float = np.finfo(np.float64).eps
    ) -> np.ndarray:

    err1:str = "A n'est pas en 2D !"
    err2:str = "A est vide !"
    err3:str = "A est rectangulaire !"
    err4:str = "Ordre de A != ordre de b."
    err5:str = "A n'est pas à diagonale dominante !"

    if a.ndim != 2:
        raise np.linalg.LinAlgError(err1)

    if a.size == 0:
        raise np.linalg.LinAlgError(err2)

    n, m = a.shape
    if n != m:
        raise np.linalg.LinAlgError(err3)

    if n != b.shape[0]:
        raise np.linalg.LinAlgError(err4)

    for i in range(n):
        diag = np.fabs(a[i,i])
        horsdiag = np.sum(np.fabs(a[i,:])) - diag
        if not np.greater_equal(diag, horsdiag):
            raise np.linalg.LinAlgError(err5)

    x0 = np.ones_like(b, dtype=np.float64)
    x1 = np.ones_like(b, dtype=np.float64)
    s = np.zeros(3, dtype=np.float64)

    it: int = 1
    while it < k:
        for i in range(n):
            s[0] = s[1] = s[2] = 0
            for j in range(i,n):
                aij = a[i,j]
                s[0] += aij * x1[j]
                s[1] += aij * x0[j]
            for j in range(i+1,n):
                s[2] += a[i,j] * x0[j]
            aii = a[i,i]
            x1[i] = x0[i] + omega * ((b[i] - s[1] - s[2]) / aii - x0[i]) + r * (s[1] - s[0]) / aii

        norme = np.linalg.norm(b - a @ x1)
        if np.less(norme, tol) == True or it == k:
            break
        else:
            x0 = x1
            it += 1

    return x1.astype(np.float64)

def systeme_lineaire_non_symetrique(n: np.uint16) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    a = rng.random(size=(n,n), dtype = np.float64)
    x = rng.random(size=n, dtype = np.float64)
    b = np.zeros(n, dtype = np.float64)

    for i in range(n):
        a[i,i] *= n*n
        for j in range(n):
            b[i] += a[i,j] * x[j]
    
    return a.astype(np.float64), \
            b.astype(np.float64), \
            x.astype(np.float64)


def test_sor(a: np.ndarray, b: np.ndarray):
    start1 = t.perf_counter()
    x1 = sor(a, b, 1.895)
    print(f"Temps SOR : {t.perf_counter()-start1} s")
    start2 = t.perf_counter()
    x2 = np.linalg.solve(a,b)
    print(f"Temps SOR NP : {t.perf_counter()-start2} s")
    print("Vérification :", end="")
    v = np.allclose(x1, x2, np.finfo(np.float64).eps)
    if v:
        print("OK")

def test_aor(a: np.ndarray, b: np.ndarray):
    start1 = t.perf_counter()
    x1 = aor(a, b, 1.28, 0.47)
    print(f"Temps AOR : {t.perf_counter()-start1} s")
    start2 = t.perf_counter()
    x2 = np.linalg.solve(a,b)
    print(f"Temps AOR NP : {t.perf_counter()-start2} s")
    print("Vérification :", end="")
    v = np.allclose(x1, x2, np.finfo(np.float64).eps)
    if v:
        print("OK")


if __name__ == "__main__":
    a, b, _ = systeme_lineaire_non_symetrique(200)
    print("--------------------------------------")
    test_sor(a, b)
    print("--------------------------------------")
    test_aor(a, b)
    print("--------------------------------------")
