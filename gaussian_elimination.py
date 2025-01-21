import numpy as np
import time as t

def pivot_gauss(
    a: np.ndarray,
    b: np.ndarray,
    tol = np.finfo(np.float64).eps
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

    for i in range(n):
        pivot = a[i,i]
        if np.isclose(np.fabs(pivot), 0.0, atol=tol) == True:
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
    x[n-1] = b[n-1] / a[n-1, n-1]
    for i in range(n-1, -1, -1):
        s = 0
        for j in range(i+1, n):
            s += a[i][j] * x[j]
        x[i] = (b[i] - s) / a[i][i]

    return x


def systeme_lineaire_non_symetrique(n: np.uint16) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    a = rng.random(size=(n,n))
    x = rng.random(size=n)
    b = np.zeros(n, dtype = np.float64)

    for i in range(n):
        for j in range(n):
            b[i] += a[i,j] * x[j]
    
    return a.astype(np.float64), b.astype(np.float64)


def test1():
	a = np.array([[3, -1, 9],
				[7, 2, -4],
				[-5, 8, 6]], dtype = np.float64)
	b = np.array([62, -13, 69], dtype = np.float64)
	x1 = pivot_gauss(a,b)
	x2 = np.linalg.solve(a,b)
	print("\nTEST 1 :")
	print("--------")
	print("X       \t=", x1)
	print("X Numpy \t=", x2)
	print("Vérification : ", end="")
	if np.allclose(x1, x2, np.finfo(np.float64).eps):
		print("OK")

def test2():
    a, b = systeme_lineaire_non_symetrique(5)
    x1 = pivot_gauss(a,b)
    x2 = np.linalg.solve(a,b)
    print("\nTEST 2 :")
    print("--------")
    print("X       \t=", x1)
    print("X Numpy \t=", x2)
    print("Vérification : ", end="")
    if np.allclose(x1, x2, np.finfo(np.float64).eps):
        print("OK")

def test3():
    a, b = systeme_lineaire_non_symetrique(200)
    start1 = t.perf_counter()
    x1 = pivot_gauss(a,b)
    print("\nTEST 3 :")
    print("--------")
    print(f"Temps 1 : {t.perf_counter()-start1} s")
    start2 = t.perf_counter()
    x2 = np.linalg.solve(a,b)
    print(f"Temps 2 : {t.perf_counter()-start2} s\n")
    print("Vérification : ", end="")
    if np.allclose(x1, x2, np.finfo(np.float64).eps):
        print("OK")


if __name__ == "__main__":
    test1()
    test2()
    test3()
