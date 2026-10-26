"""Блок 2 (дополнение). Модели на внешних библиотеках: ARIMA (statsmodels), XGBoost, LightGBM.

ВНИМАНИЕ: в среде разработки этих библиотек не было, поэтому ЭТОТ СКРИПТ НЕ ЗАПУСКАЛСЯ с реальными
библиотеками, а результатов от него в репозитории нет. Запуск: pip install -r requirements-libs.txt,
затем сначала python src/forecast.py, потом python src/forecast_libs.py.
Проверка качества и интервалы устроены так же, как в forecast.py (скользящая проверка, лог-шкала).
"""
import warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from common import SERIES, CH, RES, get_series
from forecast import H, MIN_TRAIN, N_SIM, SEED, FUT, Ti, T, _feat, metrics, simulate, naive_model

ARIMA_ORDERS = [(0, 1, 0), (1, 1, 0), (0, 1, 1)]   # SARIMA без сезонной части: сезонность в годовых данных отсутствует


# ------------------------------------------------------------------ ARIMA (statsmodels)
class ArimaModel:
    """Подбирает порядок по AIC на обучающей части; trend='t' — дрейф при d=1."""

    def __init__(self, z):
        self.z, self.res = np.asarray(z, float), None
        best_aic = np.inf
        for order in ARIMA_ORDERS:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = ARIMA(self.z, order=order, trend="t").fit()
                if np.isfinite(r.aic) and r.aic < best_aic:
                    best_aic, self.res, self.order = r.aic, r, order
            except Exception:                      # на очень коротких рядах подгонка может не сойтись
                continue

    def __call__(self, h, sim_errs=None):
        if self.res is None:                       # запасной вариант: наивный прогноз
            return np.repeat(self.z[-1], h)
        return np.asarray(self.res.forecast(h), float)

    def intervals(self, h):
        """Интервалы statsmodels в лог-шкале -> исходные единицы."""
        out = {}
        for name, alpha in (("80", .20), ("95", .05)):
            if self.res is None:
                out[name] = (np.repeat(np.nan, h), np.repeat(np.nan, h)); continue
            ci = np.asarray(self.res.get_forecast(h).conf_int(alpha=alpha))
            out[name] = (Ti(ci[:, 0]), Ti(ci[:, 1]))
        return out


def arima_model(z):
    m = ArimaModel(z)
    return lambda h, sim_errs=None: m(h)


# ------------------------------------------------------------------ XGBoost / LightGBM (те же признаки, что у sklearn-бустинга)
def make_booster(make_regressor):
    def model(z):
        X = np.array([_feat(z, t) for t in range(3, len(z))])
        y = np.array([z[t] - z[t - 1] for t in range(3, len(z))])
        reg = make_regressor().fit(X, y)

        def predict(h, sim_errs=None):
            hist, out = list(z), []
            for i in range(h):
                f = hist[-1] + float(reg.predict(np.array([_feat(hist, len(hist))]))[0])
                hist.append(f + (0.0 if sim_errs is None else sim_errs[i]))
                out.append(hist[-1])
            return np.array(out)
        return predict
    return model


xgb_model = make_booster(lambda: XGBRegressor(n_estimators=150, learning_rate=.05, max_depth=2, min_child_weight=1,
                                              subsample=1.0, random_state=SEED, verbosity=0))
lgbm_model = make_booster(lambda: LGBMRegressor(n_estimators=150, learning_rate=.05, max_depth=2, num_leaves=4,
                                                min_child_samples=2, min_data_in_bin=1, random_state=SEED, verbose=-1))

MODELS = {"ARIMA (statsmodels)": arima_model, "XGBoost (лаги)": xgb_model, "LightGBM (лаги)": lgbm_model}
KIND = {"ARIMA (statsmodels)": "статистическая", "XGBoost (лаги)": "машинное обучение", "LightGBM (лаги)": "машинное обучение"}


def rolling_cv(models, z, imputed):
    res = {m: [] for m in models}
    for t in range(MIN_TRAIN, len(z)):
        if imputed[t]: continue
        for m, f in models.items():
            res[m].append((t, z[t], f(z[:t])(1)[0]))
    return res


if __name__ == "__main__":
    met_rows, fc_rows, store = [], [], {}
    for key, (_, title, unit) in SERIES.items():
        s, imputed = get_series(key); z = T(s.to_numpy()); imp = imputed.to_numpy()
        cv = rolling_cv(MODELS, z, imp); errs = {}
        for m in MODELS:
            mt, e = metrics(cv[m]); errs[m] = e
            met_rows.append({"series": key, "model": m, "approach": KIND[m], **mt})
        for m, fn in MODELS.items():
            point = Ti(fn(z)(H))
            if m.startswith("ARIMA"):
                iv = ArimaModel(z).intervals(H); lo80, hi80 = iv["80"]; lo95, hi95 = iv["95"]
            else:
                sims = simulate(fn, z, errs[m])
                lo80, hi80, lo95, hi95 = (np.percentile(sims, p, axis=0) for p in (10, 90, 2.5, 97.5))
            for i, yr in enumerate(FUT):
                fc_rows.append({"series": key, "model": m, "year": yr, "forecast": point[i],
                                "lo80": min(lo80[i], point[i]), "hi80": max(hi80[i], point[i]),
                                "lo95": min(lo95[i], point[i]), "hi95": max(hi95[i], point[i])})
        store[key] = s
    ML = pd.DataFrame(met_rows); FL = pd.DataFrame(fc_rows)
    ML.round(2).to_csv(RES / "libs_model_metrics.csv", index=False, encoding="utf-8-sig")
    FL.round(2).to_csv(RES / "libs_forecast_2022_2024.csv", index=False, encoding="utf-8-sig")

    # общая таблица: модели из forecast.py + модели на библиотеках; лучшая — по RMSE (эталон в выбор не входит)
    base = pd.read_csv(RES / "model_metrics.csv").drop(columns=["best"])
    ALL = pd.concat([base, ML], ignore_index=True); ALL["best"] = False
    for key in SERIES:
        sub = ALL[(ALL.series == key) & (ALL.approach != "эталон")]; ALL.loc[sub.RMSE.idxmin(), "best"] = True
    ALL.round(2).to_csv(RES / "all_models_metrics.csv", index=False, encoding="utf-8-sig")
    print(ALL.round(1).to_string())

    # график: прогнозы моделей на библиотеках
    fig, axs = plt.subplots(2, 2, figsize=(14, 8.5)); colors = {"ARIMA (statsmodels)": "#1f77b4", "XGBoost (лаги)": "#2ca02c", "LightGBM (лаги)": "#ff7f0e"}
    for ax, key in zip(axs.ravel(), ["import_qty", "import_value", "export_qty", "export_value"]):
        s = store[key]; ax.plot(s.index, s, "o-", color="k", label="факт")
        a = FL[(FL.series == key) & FL.model.str.startswith("ARIMA")]
        ax.fill_between(a.year, a.lo95, a.hi95, color=colors["ARIMA (statsmodels)"], alpha=.15, label="ARIMA 95%")
        for m in MODELS:
            f = FL[(FL.series == key) & (FL.model == m)]
            ax.plot([s.index[-1], *f.year], [s.iloc[-1], *f.forecast], "o--", color=colors[m], label=m)
        ax.axvline(2021.5, color="grey", ls=":"); ax.set_title(f"{SERIES[key][1]}, {SERIES[key][2]}")
        ax.set_ylim(0, min(a.hi95.max(), 3 * s.max()) * 1.05); ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Арбузы (080711): прогноз моделей на внешних библиотеках, 2022–2024", y=1.0); fig.tight_layout()
    fig.savefig(CH / "06_libs_forecasts.png", bbox_inches="tight"); plt.close(fig)
