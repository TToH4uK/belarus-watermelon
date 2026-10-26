"""Блок 1. Декомпозиция рядов: тренд (фильтр Ходрика–Прескотта), цикл, остаток; сезонность не определима."""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from common import SERIES, CH, RES, get_series

LAMBDA = 6.25   # параметр сглаживания для годовых данных (Ravn–Uhlig)


def hp_trend(y: np.ndarray, lam: float = LAMBDA) -> np.ndarray:
    n = len(y)
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i:i + 3] = [1, -2, 1]
    return np.linalg.solve(np.eye(n) + lam * D.T @ D, y)


def decompose(s: pd.Series) -> pd.DataFrame:
    trend = hp_trend(s.to_numpy())
    detr = s.to_numpy() - trend
    cycle = pd.Series(detr, index=s.index).rolling(3, center=True, min_periods=2).mean()
    return pd.DataFrame({"observed": s, "trend": trend, "seasonal": 0.0, "cycle": cycle,
                         "residual": detr - cycle.to_numpy()})


if __name__ == "__main__":
    out = []
    for key, (_, title, unit) in SERIES.items():
        s, imputed = get_series(key)
        d = decompose(s)
        d.insert(0, "series", key); out.append(d.reset_index())
        fig, axs = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
        axs[0].plot(d.index, d.observed, "o-", color="#1f77b4", label="наблюдение")
        axs[0].plot(d.index, d.trend, color="#d62728", lw=2, label="тренд (HP, λ=6,25)")
        if imputed.any(): axs[0].plot(d.index[imputed], d.observed[imputed], "o", mfc="w", color="k", label="интерполяция")
        axs[0].legend(fontsize=8); axs[0].set_title(f"{title} арбузов (080711): декомпозиция, {unit}")
        axs[1].bar(d.index, d.cycle, color="#2ca02c"); axs[1].axhline(0, color="k", lw=.7); axs[1].set_ylabel("цикл")
        axs[2].bar(d.index, d.residual, color="#7f7f7f"); axs[2].axhline(0, color="k", lw=.7); axs[2].set_ylabel("остаток")
        axs[3].plot(d.index, d.seasonal, color="#9467bd"); axs[3].set_ylabel("сезонность")
        axs[3].text(.5, .55, "не определима: данные годовые (период сезонности = 1)", transform=axs[3].transAxes, ha="center", fontsize=9)
        axs[3].set_ylim(-1, 1)
        fig.tight_layout(); fig.savefig(CH / f"02_decomposition_{key}.png", bbox_inches="tight"); plt.close(fig)
    full = pd.concat(out); full.round(3).to_csv(RES / "decomposition.csv", index=False, encoding="utf-8-sig")
    # доля дисперсии, объяснённая компонентами
    for key in SERIES:
        d = full[full.series == key]; v = d.observed.var()
        print(key, "var share trend/cycle/resid:", round(d.trend.var() / v, 2), round(d.cycle.var() / v, 2), round(d.residual.var() / v, 2),
              "| corr(observed,trend)=", round(np.corrcoef(d.observed, d.trend)[0, 1], 2))
