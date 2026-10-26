"""Блок 3. Курс BYN/USD (среднегодовой официальный, 2008–2024) и его связь со стоимостью импорта/экспорта арбузов."""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from common import CH, RES, load_fx, load_raw, FIRST_IMPORT_YEAR

fx, raw = load_fx(), load_raw()
rate = fx.rate_byn

# ---- график курса
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(fx.index, rate, "o-", color="#2ca02c")
ax.axvline(2015.5, color="grey", ls=":"); ax.text(2015.6, rate.max() * .9, "деноминация 2016 г.\n(10 000 BYR = 1 BYN)", fontsize=8)
ax.set_title("Среднегодовой официальный курс, BYN за 1 долл. США (2008–2015 пересчитаны из BYR)"); ax.set_ylabel("BYN/USD")
fig.tight_layout(); fig.savefig(CH / "04_usd_byn_rate.png", bbox_inches="tight"); plt.close(fig)

# ---- корреляции (наблюдаемые годы, интерполированные точки не используются)
rows = []
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
for j, (flow, col, color) in enumerate([("Импорт", "import_value_kusd", "#d62728"), ("Экспорт", "export_value_kusd", "#1f77b4")]):
    v = raw[col].dropna()
    if flow == "Импорт": v = v.loc[FIRST_IMPORT_YEAR:]
    r = rate.loc[v.index]
    df = pd.DataFrame({"value": v, "rate": r})
    d1 = pd.DataFrame({"value": np.log1p(v).diff(), "rate": np.log(r).diff()})
    # разности считаем только между соседними годами
    d1 = d1[(d1.index.to_series().diff() == 1)].dropna()
    for name, d in [("уровни", df), ("годовые изменения (лог)", d1)]:
        pr, pp = pearsonr(d.value, d.rate); sr, sp = spearmanr(d.value, d.rate)
        rows.append({"поток": flow, "переменные": name, "n": len(d), "Пирсон r": pr, "p (Пирсон)": pp, "Спирмен ρ": sr, "p (Спирмен)": sp})
    ax = axs[0, j]; ax.scatter(df.rate, df.value, color=color)
    for y, (x, yv) in df.iterrows(): ax.annotate(str(y), (x, yv), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("BYN за 1 долл. США"); ax.set_ylabel("стоимость, тыс. долл."); ax.set_title(f"{flow}: стоимость и курс (уровни)")
    ax2 = axs[1, j]; ax2.bar(v.index, v, color=color, alpha=.6); ax2.set_ylabel("стоимость, тыс. долл.")
    ax3 = ax2.twinx(); ax3.plot(rate.index, rate, "k.-"); ax3.set_ylabel("BYN/USD"); ax3.grid(False)
    ax2.set_title(f"{flow}: стоимость (столбцы) и курс (линия)")
fig.tight_layout(); fig.savefig(CH / "05_correlation.png", bbox_inches="tight"); plt.close(fig)
C = pd.DataFrame(rows); C.round(3).to_csv(RES / "correlation.csv", index=False, encoding="utf-8-sig")
fx.to_csv(RES / "usd_byn_annual_used.csv", encoding="utf-8-sig")
print(C.round(3).to_string())
