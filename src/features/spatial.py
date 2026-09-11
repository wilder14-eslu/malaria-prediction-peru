"""Features espaciales: senal de casos en distritos vecinos.

Usa la tabla de adyacencia precomputada por
``pipelines/reference/build_district_adjacency.py``
(``data/reference/district_adjacency.csv``), no calcula geometria en runtime.

Motivacion (ver docs/architecture.md, seccion "Propuesta de rediseno"):
la malaria se propaga entre distritos colindantes; un modelo por-distrito
aislado no puede aprender ese patron. Etapa 1 del rediseno: agregar el
numero de casos de los distritos vecinos, en la semana anterior (lag 1,
para no filtrar informacion futura), como feature adicional de LightGBM/XGBoost.
"""

from __future__ import annotations

import pandas as pd


def load_adjacency(path: str = "data/reference/district_adjacency.csv") -> pd.DataFrame:
    return pd.read_csv(path, dtype={"ubigeo": str, "neighbor_ubigeo": str})


def add_neighbor_lag_features(
    df: pd.DataFrame,
    adjacency: pd.DataFrame,
    target_col: str = "cases_total",
    lag: int = 1,
) -> pd.DataFrame:
    """Agrega ``neighbor_cases_lag_{lag}`` (suma de ``target_col`` en distritos
    vecinos, ``lag`` semanas atras) y ``neighbor_count`` (cuantos vecinos
    tenian dato disponible esa semana, para distinguir "0 casos confirmados
    en los vecinos" de "no hay datos de los vecinos").

    Requiere que ``df`` ya haya pasado por ``complete_weekly_grid`` (columna
    ``epi_index`` continua), igual que los lags temporales.
    """
    lookup = df[["ubigeo", "epi_index", target_col]].rename(
        columns={"ubigeo": "neighbor_ubigeo", target_col: "neighbor_value"}
    )

    expanded = df[["ubigeo", "epi_index"]].merge(adjacency, on="ubigeo", how="left")
    expanded["target_epi_index"] = expanded["epi_index"] - lag

    joined = expanded.merge(
        lookup,
        left_on=["neighbor_ubigeo", "target_epi_index"],
        right_on=["neighbor_ubigeo", "epi_index"],
        how="left",
        suffixes=("", "_neighbor"),
    )

    agg = joined.groupby(["ubigeo", "epi_index"]).agg(
        **{
            f"neighbor_cases_lag_{lag}": ("neighbor_value", "sum"),
            "neighbor_count": ("neighbor_value", "count"),
        }
    )
    # Si un distrito no tiene ningun vecino con dato disponible esa semana
    # (neighbor_count == 0), la suma no representa "0 casos confirmados"
    # sino "sin senal": se deja NaN en vez de 0, igual que growth_rate/ratios.
    agg.loc[agg["neighbor_count"] == 0, f"neighbor_cases_lag_{lag}"] = pd.NA

    return df.merge(agg, on=["ubigeo", "epi_index"], how="left")
