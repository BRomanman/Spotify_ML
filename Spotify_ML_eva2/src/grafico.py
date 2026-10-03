from pathlib import Path
import matplotlib.pyplot as plt

def _save(fig, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path

def save_histogram(series, title, xlabel, ylabel, path, bins=20):
    fig, ax = plt.subplots(figsize=(10,5)); ax.hist(series, bins=bins); ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); fig.tight_layout(); p=_save(fig,path); plt.close(fig); return p

def save_barh(labels, values, title, xlabel, path):
    fig, ax = plt.subplots(figsize=(10,7)); ax.barh(labels, values); ax.set_title(title); ax.set_xlabel(xlabel); fig.tight_layout(); p=_save(fig,path); plt.close(fig); return p

def save_metrics_plot(results, path):
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    axes[0].bar(results["Modelo"], results["MAE"]); axes[0].set_title("MAE")
    axes[1].bar(results["Modelo"], results["RMSE"]); axes[1].set_title("RMSE")
    axes[2].bar(results["Modelo"], results["R2"]); axes[2].set_title("R2")
    fig.tight_layout(); p=_save(fig,path); plt.close(fig); return p

def save_scatter(y_true, y_pred, title, xlabel, ylabel, path):
    fig, ax = plt.subplots(figsize=(8,6)); ax.scatter(y_true, y_pred, alpha=0.35); ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); fig.tight_layout(); p=_save(fig,path); plt.close(fig); return p
