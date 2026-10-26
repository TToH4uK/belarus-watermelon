"""Общие функции: загрузка данных по арбузам (ТН ВЭД 080711) и курсов BYN/USD."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl

ROOT = Path(__file__).resolve().parent.parent
DATA, CH, RES = ROOT / "data", ROOT / "charts", ROOT / "results"
CH.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
mpl.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})

# 4 показателя: (ключ) -> (колонка, подпись, ед. изм.)
SERIES = {
    "import_qty":   ("import_qty_t",      "Импорт, количество",  "т"),
    "import_value": ("import_value_kusd", "Импорт, стоимость",   "тыс. долл. США"),
    "export_qty":   ("export_qty_t",      "Экспорт, количество", "т"),
    "export_value": ("export_value_kusd", "Экспорт, стоимость",  "тыс. долл. США"),
}
FIRST_IMPORT_YEAR = 2010   # данных по импорту за 2008–2009 нет


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA / "watermelon_080711_annual.csv").set_index("year")


def get_series(key: str):
    """Возвращает (ряд для моделирования, булева маска 'значение восстановлено интерполяцией').
    Импорт: ряд начинается с 2010 г.; пропуск 2012–2013 заполнен лог-линейной интерполяцией."""
    col = SERIES[key][0]
    s = load_raw()[col].astype(float)
    imputed = pd.Series(False, index=s.index)
    if key.startswith("import"):
        s = s.loc[FIRST_IMPORT_YEAR:]
        imputed = imputed.loc[FIRST_IMPORT_YEAR:]
        miss = s.isna()
        s = np.exp(np.log(s).interpolate(limit_area="inside"))
        imputed = miss
    return s, imputed


def load_fx() -> pd.DataFrame:
    return pd.read_csv(DATA / "usd_byn_annual.csv").set_index("year")
