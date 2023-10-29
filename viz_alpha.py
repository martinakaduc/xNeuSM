import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tueplots import bundles

plt.rcParams.update(bundles.icml2022())
sns.set_style("darkgrid", rc=bundles.icml2022())

if __name__ == "__main__":
    k = np.linspace(0, 12, 13)
    list_alpha = np.linspace(0.1, 0.9, 9)

    def fn(x, k):
        return (1 - x) ** (k+1)

    for alpha in list_alpha:
        y = fn(alpha, k)
        plt.plot(
            k, y, label=f"$\\alpha={round(alpha, 1)}$", marker="o", markersize=2)

    plt.legend(ncol=2)
    plt.xlabel("$K$")
    plt.ylabel("$(1-\\alpha)^{K+1}$")
    plt.tight_layout()
    plt.savefig("alpha.pdf")
