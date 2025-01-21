from math import comb
import numpy as np
from functools import cache
import matplotlib.pyplot as mpl
import time

@cache
def bernstein(
    n: np.uint8, m: np.uint8,
    i: np.uint8, j: np.uint8,
    u: np.float32, v: np.float32
    ) -> np.float32:

    bu = comb(n, i) * u**i * (1 - u)**(n - i)
    bv = comb(m, j) * v**j * (1 - v)**(m - j)
    
    return bu * bv


def rbezier(
    u: np.float32, v: np.float32,
    points: np.ndarray, poids: np.ndarray
    ) -> np.ndarray:

    n = points.shape[1]
    m = points.shape[0]

    xyz = np.zeros(3, dtype=np.float32)
    b = np.float32(0)
    rationnelle = np.float32(0)

    for i in range(n):
        for j in range(m):
            pij = poids[i, j]
            b = bernstein(n, m, i, j, u, v) * pij
            for k in range(3):
                xyz[k] += points[i, j, k] * b
            rationnelle += b

    return np.divide(xyz, rationnelle, dtype=np.float64)


def afficher(
    points: np.ndarray, poids: np.ndarray,
    num_points: int = 50+1
    ) -> None:

    a = time.time()

    fig = mpl.figure(figsize=(8,8))
    ax = fig.add_subplot(projection="3d")

    us = np.linspace(0, 1, num_points, dtype=np.float32)
    vs = np.linspace(0, 1, num_points, dtype=np.float32)
    uvs = np.r_[[us, vs]]

    for u in us:
        tv_ = tuple(rbezier(u, v, points, poids) for v in uvs[0])
        tv = np.asarray(tv_)
        iso = np.array(tv, dtype=np.float32)
        ax.plot(iso[:, 0], iso[:, 1], iso[:, 2], linestyle="-",
                linewidth=0.5, color="gray", antialiased=True)

    for v in vs:
        tu_ = tuple(rbezier(u, v, points, poids) for u in uvs[1])
        tu = np.asarray(tu_)
        iso = np.array(tu, dtype=np.float32)
        ax.plot(iso[:, 0], iso[:, 1], iso[:, 2], linestyle="-",
                linewidth=0.5, color="gray", antialiased=True)

    pc0, pc1, pc2 = points[:, :, 0], points[:, :, 1], points[:, :, 2]
    
    ax.scatter(pc0.ravel(), pc1.ravel(), pc2.ravel(),
                color="k", edgecolor="k", s=50)
    
    n = points.shape[1]
    m = points.shape[0]

    tuple(ax.plot(pc0[i, :], pc1[i, :], pc2[i, :], linestyle="-", dashes=(2,4),
            alpha=0.75, linewidth=0.5, color="k") for i in range(m))

    tuple(ax.plot(pc0[:, j], pc1[:, j], pc2[:, j], linestyle="-", dashes=(2,4),
            alpha=0.75, linewidth=0.5, color="k") for j in range(n))

    b = time.time()
    print("temps :", str(b-a), "s")

    fig.tight_layout()
    ax.set_axis_off()
    ax.view_init(elev=30, azim=30)
    ax.set_box_aspect(None, zoom=1.25)
    mpl.show()


if __name__ == "__main__":
    points = np.array([
        [[-15, 0, 600], [5, 0, 800], [25, 0, 800], [45, 0, 500]],
        [[-15, 1, 700], [5, 1, 500], [25, 1, 400], [45, 1, 700]],
        [[-15, 2, 400], [5, 2, 400], [25, 2, 500], [45, 2, 300]],
        [[-15, 3, 400], [5, 3, 700], [25, 3, 600], [45, 3, 300]]
    ], dtype=np.float32)

    w = 20
    poids = np.array([
        [1, w, w, 1],
        [w, w*4, w*4, w],
        [w, w*4, w*4, w],
        [1, w, w, 1]
    ], dtype=np.float32)

    afficher(points, poids)
