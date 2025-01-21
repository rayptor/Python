import numpy as np

def cholesky(
    a: np.array,
	b: np.ndarray,
	) -> np.ndarray:

	print("Vérification de la matrice A... ", end="")

	if a.ndim != 2:
		raise np.linalg.LinAlgError("A n'est pas en 2D !")

	if a.size == 0:
		raise np.linalg.LinAlgError("A est vide !")

	n, m = a.shape
	if n != m:
		raise np.linalg.LinAlgError("A est rectangulaire !")
	
	valp = np.linalg.eigvals(a)
	if np.allclose(a, a.T) == False and np.any(valp) <= 0.0:
		raise np.linalg.LinAlgError("A n'est SDP")

	if (n != b.shape[0]):
		raise np.linalg.LinAlgError("Ordre de A != ordre de b.")
    
	print("OK")

	l = a.copy()

	for i in range(n):
		tmp = l[i,i]
		l[i,i] = np.sqrt(tmp)
		for j in range(i+1,n):
			l[j,i] /= l[i,i]
		for k in range(i+1,n):
			l[i,k] = 0
			for j in range(k,n):
				l[j,k] -= l[j,i] * l[k,i]

	x = np.zeros(n)
	y = np.zeros(n)

	# Ly = b
	y[0] = b[0] / l[0,0]
	for i in range(n):
		s = 0.0
		for j in range(i):
			s += l[i,j] * y[j]
		y[i] = (b[i] - s) / l[i,i]

	# L^Tx = y
	lt = l.T
	x[n-1] = y[n-1] / lt[n-1,n-1]
	s = 0.0
	for i in range(n-1,-1,-1):
		s = y[i]
		for j in range(i+1,n):
			s -= lt[i,j]*x[j]
		x[i] = s / lt[i,i]

	return l, x

if __name__ == "__main__":
	a = np.array(
	[[ 9,  3,  6, 9],
	[ 3,  5,  8,  7],
	[ 6,  8, 14, 13],
	[ 9,  7, 13, 63]])
	b = np.array([66,  54,  99, 252])
	l, x = cholesky(a,b)
	print(f"{l=}")
	print(f"{x=}")

