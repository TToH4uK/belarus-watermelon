"""Блок 2. Прогноз на 2022–2024: статистическая модель (ETS: затухающий тренд Хольта) и модель машинного
обучения (градиентный бустинг по лагам). Выбор лучшей — по метрикам скользящей проверки (rolling origin)."""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor
from common import SERIES, CH, RES, get_series, load_raw

H, MIN_TRAIN, N_SIM, SEED = 3, 6, 1500, 42
FUT = [2022, 2023, 2024]
rng = np.random.default_rng(SEED)
T = np.log1p                       # лог-преобразование (в рядах есть нули)
Ti = lambda z: np.clip(np.expm1(z), 0, None)


# ------------------------------------------------------------------ статистическая модель: ETS(A,Ad,N)
def _ets_run(y, a, b, phi):
    l, tr, errs = y[0], y[1] - y[0], []
    for t in range(1, len(y)):
        f = l + phi * tr
        errs.append(y[t] - f)
        l_new = a * y[t] + (1 - a) * f
        tr = b * (l_new - l) + (1 - b) * phi * tr
        l = l_new
    return l, tr, np.array(errs)


def ets_fit(y):
    best = None
    for a0 in (.3, .6, .9):
        for b0 in (.1, .4):
            r = minimize(lambda p: np.sum(_ets_run(y, *p)[2] ** 2), [a0, b0, .9],
                         bounds=[(.05, .99), (.01, .99), (.8, .98)], method="L-BFGS-B")
            if best is None or r.fun < best.fun: best = r
    return best.x


def ets_model(y):
    a, b, phi = ets_fit(y)
    l0, tr0, _ = _ets_run(y, a, b, phi)

    def predict(h, sim_errs=None):
        l, tr, out = l0, tr0, []
        for i in range(h):
            f = l + phi * tr
            obs = f + (0.0 if sim_errs is None else sim_errs[i])
            l_new = a * obs + (1 - a) * f
            tr = b * (l_new - l) + (1 - b) * phi * tr
            l = l_new; out.append(obs)   # без шума obs == f
        return np.array(out)
    return predict


# ------------------------------------------------------------------ модель машинного обучения: GradientBoosting
def _feat(z, t):   # признаки для предсказания z[t]: уровень z[t-1] и два последних прироста
    return [z[t - 1], z[t - 1] - z[t - 2], z[t - 2] - z[t - 3]]


def gb_make(z):
    X = np.array([_feat(z, t) for t in range(3, len(z))]); yv = np.array([z[t] - z[t - 1] for t in range(3, len(z))])
    return GradientBoostingRegressor(n_estimators=150, learning_rate=.05, max_depth=2, min_samples_leaf=2,
                                     subsample=1.0, random_state=SEED).fit(X, yv)


def gb_model(z):
    m = gb_make(z)

    def predict(h, sim_errs=None):
        hist = list(z); out = []
        for i in range(h):
            f = hist[-1] + m.predict([_feat(hist, len(hist))])[0]
            hist.append(f + (0.0 if sim_errs is None else sim_errs[i]))
            out.append(hist[-1])   # без шума значение == f
        return np.array(out)
    return predict


def naive_model(z):   # эталон для сравнения: «как в последнем наблюдении»
    return lambda h, sim_errs=None: np.repeat(z[-1], h)


MODELS = {"ETS (затухающий тренд)": ets_model, "GradientBoosting (лаги)": gb_model, "Наивный (эталон)": naive_model}
KIND = {"ETS (затухающий тренд)": "статистическая", "GradientBoosting (лаги)": "машинное обучение", "Наивный (эталон)": "эталон"}


def rolling_cv(z, imputed):
    """Одношаговая проверка со скользящим началом: обучаем на z[:t], прогнозируем z[t]."""
    res = {m: [] for m in MODELS}
    for t in range(MIN_TRAIN, len(z)):
        if imputed[t]: continue                               # интерполированные точки не оцениваем
        for m, f in MODELS.items():
            res[m].append((t, z[t], f(z[:t])(1)[0]))
    return res


def metrics(rows):
    t, act, pred = map(np.array, zip(*rows))
    a, p = Ti(act), Ti(pred)
    nz = a > 0
    return {"MAE": np.mean(np.abs(a - p)), "RMSE": np.sqrt(np.mean((a - p) ** 2)),
            "MAPE, %": np.mean(np.abs((a[nz] - p[nz]) / a[nz])) * 100, "n_folds": len(a)}, act - pred


def simulate(fn, z, errs):
    sigma = np.sqrt(np.mean(errs ** 2)); pred = fn(z)       # σ — RMS одношаговых ошибок скользящей проверки (лог-шкала)
    sims = np.array([pred(H, rng.normal(0, sigma, H)) for _ in range(N_SIM)])
    return Ti(sims)


if __name__ == "__main__":
    met_rows, fc_rows, store = [], [], {}
    for key, (_, title, unit) in SERIES.items():
        s, imputed = get_series(key); years = s.index.to_numpy()
        z = T(s.to_numpy()); imp = imputed.to_numpy()
        cv = rolling_cv(z, imp)
        errs = {}
        for m in MODELS:
            mt, e = metrics(cv[m]); errs[m] = e
            met_rows.append({"series": key, "model": m, "approach": KIND[m], **mt})
        mm = pd.DataFrame([r for r in met_rows if r["series"] == key and r["approach"] != "эталон"])
        best = mm.loc[mm.RMSE.idxmin(), "model"]
        store[key] = {"s": s, "best": best}
        for m in [x for x in MODELS if x != "Наивный (эталон)"]:
            point = Ti(MODELS[m](z)(H))
            sims = simulate(MODELS[m], z, errs[m])
            q = lambda p: np.percentile(sims, p, axis=0)
            for i, yr in enumerate(FUT):
                fc_rows.append({"series": key, "model": m, "best": m == best, "year": yr, "forecast": point[i],
                                "lo80": min(q(10)[i], point[i]), "hi80": max(q(90)[i], point[i]),
                                "lo95": min(q(2.5)[i], point[i]), "hi95": max(q(97.5)[i], point[i])})
    M = pd.DataFrame(met_rows); M["best"] = False
    for key in SERIES:
        sub = M[(M.series == key) & (M.approach != "эталон")]; M.loc[sub.RMSE.idxmin(), "best"] = True
    M.round(2).to_csv(RES / "model_metrics.csv", index=False, encoding="utf-8-sig")
    F = pd.DataFrame(fc_rows); F.round(2).to_csv(RES / "forecast_2022_2024.csv", index=False, encoding="utf-8-sig")
    print(M.round(2).to_string()); print(F[F.best].round(1).to_string())

    # ---- график итогового прогноза
    fig, axs = plt.subplots(2, 2, figsize=(14, 8.5))
    for ax, key in zip(axs.ravel(), ["import_qty", "import_value", "export_qty", "export_value"]):
        _, title, unit = SERIES[key]; s = store[key]["s"]; _, imputed = get_series(key)
        ax.plot(s.index, s, "o-", color="k", label="факт")
        if imputed.any(): ax.plot(s.index[imputed], s[imputed], "o", mfc="w", color="k", label="интерполяция")
        fb = F[(F.series == key) & F.best]; fo = F[(F.series == key) & ~F.best]
        ax.fill_between(fb.year, fb.lo95, fb.hi95, color="#1f77b4", alpha=.15, label="95% интервал")
        ax.fill_between(fb.year, fb.lo80, fb.hi80, color="#1f77b4", alpha=.30, label="80% интервал")
        ax.plot([s.index[-1], *fb.year], [s.iloc[-1], *fb.forecast], "o-", color="#1f77b4", lw=2, label=f"прогноз: {store[key]['best']}")
        ax.plot([s.index[-1], *fo.year], [s.iloc[-1], *fo.forecast], "s--", color="#ff7f0e", lw=1.3, label=f"альтернатива: {fo.model.iloc[0]}")
        ax.axvline(2021.5, color="grey", ls=":"); ax.set_title(f"{title}, {unit}"); ax.set_ylim(0, min(fb.hi95.max(), 3 * s.max()) * 1.05); ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Арбузы свежие (080711): прогноз на 2022–2024 гг.", y=1.0, fontsize=12); fig.tight_layout()
    fig.savefig(CH / "03_forecasts.png", bbox_inches="tight"); plt.close(fig)

    # ---- график исходных данных
    raw = load_raw(); fig, axs = plt.subplots(2, 2, figsize=(13, 7))
    for ax, key in zip(axs.ravel(), ["import_qty", "import_value", "export_qty", "export_value"]):
        col, title, unit = SERIES[key]; ax.bar(raw.index, raw[col], color="#1f77b4" if "export" in key else "#d62728")
        ax.set_title(f"{title}, {unit}")
        if key.startswith("import"):
            ax.text(.02, .92, "нет данных: 2008–2009, 2012–2013", transform=ax.transAxes, fontsize=8, va="top", color="dimgrey")
    fig.suptitle("Арбузы свежие (080711): исходные данные Белстата, 2008–2021", y=1.0); fig.tight_layout()
    fig.savefig(CH / "01_raw_data.png", bbox_inches="tight"); plt.close(fig)
