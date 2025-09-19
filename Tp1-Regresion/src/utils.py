
import numpy as np
import matplotlib.pyplot as plt
from src.metrics import mse

# -----------------------------
# 1) Barras: ECM train vs val
# -----------------------------
def plot_bar_mse(results, title="Comparación ECM por modelo"):
    labels, train_ecm, val_ecm = [], [], []
    for name, d in results.items():
        labels.append(name)
        train_ecm.append(mse(d["y_train"], d["yhat_train"]))
        val_ecm.append(mse(d["y_val"], d["yhat_val"]))

    x = np.arange(len(labels)); width = 0.35
    plt.figure(figsize=(8,6))
    plt.bar(x - width/2, train_ecm, width, label="Train ECM")
    plt.bar(x + width/2, val_ecm,   width, label="Val ECM")
    plt.xticks(x, labels)
    plt.ylabel("ECM")
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    for i, v in enumerate(train_ecm):
        plt.text(x[i]-width/2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9, rotation=90)
    for i, v in enumerate(val_ecm):
        plt.text(x[i]+width/2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9, rotation=90)
    plt.show()

# -----------------------------------------
# 2) Parity plot (y_true vs y_pred) en val
# -----------------------------------------
def plot_parity(results, title="Parity plot (validación)"):
    n = len(results)
    cols = min(3, n); rows = int(np.ceil(n/cols))
    plt.figure(figsize=(5*cols, 4.5*rows))
    for i, (name, d) in enumerate(results.items(), 1):
        y = np.asarray(d["y_val"]); yhat = np.asarray(d["yhat_val"])
        lims = [min(y.min(), yhat.min()), max(y.max(), yhat.max())]
        plt.subplot(rows, cols, i)
        plt.scatter(y, yhat, s=12, alpha=0.6)
        plt.plot(lims, lims, 'k--', lw=1)  # línea y=x
        plt.xlabel("y verdadero"); plt.ylabel("y predicho")
        plt.title(f"{name} | ECM={mse(y, yhat):.2f}")
        plt.xlim(lims); plt.ylim(lims); plt.grid(alpha=0.3)
    plt.suptitle(title, y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()

# --------------------------------
# 3) Distribución de residuales
# --------------------------------
def plot_residuals(results, which="val", bins=30, title="Distribución de residuales"):
    # which: "train" o "val"
    plt.figure(figsize=(8,6))
    for name, d in results.items():
        y = np.asarray(d[f"y_{which}"]); yhat = np.asarray(d[f"yhat_{which}"])
        res = y - yhat
        # hist densidad para comparar formas
        plt.hist(res, bins=bins, histtype="step", density=True, label=name, alpha=0.9)
    plt.axvline(0, color="k", lw=1, ls="--")
    plt.title(f"{title} ({which})")
    plt.xlabel("Residual (y - y_pred)")
    plt.ylabel("Densidad")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# -----------------------------------------
# 4) Boxplot de errores absolutos en val
# -----------------------------------------
def plot_abs_error_box(results, which="val", title="Errores absolutos por modelo"):
    data = []
    labels = []
    for name, d in results.items():
        y = np.asarray(d[f"y_{which}"]); yhat = np.asarray(d[f"yhat_{which}"])
        data.append(np.abs(y - yhat))
        labels.append(name)
    plt.figure(figsize=(8,6))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("|y - y_pred|")
    plt.title(f"{title} ({which})")
    plt.grid(axis="y", alpha=0.3)
    plt.show()

# -----------------------------------------
# 5) Curva ECM por cuantil del precio (val)
#    (mide si el error cambia con el rango)
# -----------------------------------------
def plot_ecm_by_quantile(results, q=10, title="ECM por decil de precio (validación)"):
    plt.figure(figsize=(9,6))
    # cuantiles comunes según y_val global
    all_y = np.concatenate([np.asarray(d["y_val"]) for d in results.values()])
    qs = np.quantile(all_y, np.linspace(0,1,q+1))
    centers = 0.5*(qs[:-1] + qs[1:])

    for name, d in results.items():
        y = np.asarray(d["y_val"]); yhat = np.asarray(d["yhat_val"])
        ecm_bins = []
        for i in range(q):
            mask = (y >= qs[i]) & (y <= qs[i+1])
            if mask.any():
                ecm_bins.append(mse(y[mask], yhat[mask]))
            else:
                ecm_bins.append(np.nan)
        plt.plot(centers, ecm_bins, marker="o", label=name)

    plt.title(title)
    plt.xlabel("Precio (centro del decil)")
    plt.ylabel("ECM")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


