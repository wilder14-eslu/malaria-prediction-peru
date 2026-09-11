"""Carga de las dos fuentes crudas de datos abiertos de vigilancia de malaria."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.validation.schemas import LegacyWeeklySchema, LineListSchema

DISTRICT_NAMES_PATH = Path("data/reference/district_names.csv")

# Rango biologicamente plausible de edad en anios. La fuente real trae un
# puñado de registros corruptos (ej. 90130297) -- se confirmo con los datos
# reales del portal (44 de 597142 filas, 0.007%), casi seguro un valor de
# otra columna pegado por error en la exportacion. Se limpian a NaN en vez
# de descartar la fila entera, porque el resto del registro (caso, semana,
# ubigeo) sigue siendo valido y usable.
MIN_EDAD = 0
MAX_EDAD = 120


def _fill_missing_district_names(
    df: pd.DataFrame, reference_path: Path = DISTRICT_NAMES_PATH
) -> pd.DataFrame:
    """Rellena departamento/provincia/distrito faltantes usando el UBIGEO.

    La fuente legacy (2000-2008) trae, para un puñado de registros, el
    UBIGEO correcto pero los nombres vacios (confirmado con los datos
    reales: 473 de 106355 filas, ambas del UBIGEO 160109/160114 en
    Loreto). En vez de descartar esos casos reales de malaria, se
    completan los nombres a partir de la tabla de referencia de distritos
    (``pipelines/reference/build_district_names.py``, derivada de los
    limites distritales de INEI).
    """
    missing_mask = df["departamento"].isna()
    if not missing_mask.any():
        return df
    if not reference_path.exists():
        # Sin la tabla de referencia no se puede rellenar; se deja como
        # esta y la validacion de Pandera fallara con un mensaje claro
        # senalando las filas con departamento/provincia/distrito nulos.
        return df

    names = pd.read_csv(reference_path, dtype=str).set_index("ubigeo")
    df = df.copy()
    for col in ("departamento", "provincia", "distrito"):
        # astype(object) primero: si la columna llega vacia por completo
        # (todo NaN, como puede pasar en un CSV chico de prueba), pandas la
        # tipa como float64 y asignarle strings mas adelante revienta con
        # un TypeError de dtype incompatible.
        df[col] = df[col].astype(object)
        lookup = df.loc[missing_mask, "ubigeo"].map(names[col])
        df.loc[missing_mask, col] = df.loc[missing_mask, col].fillna(lookup)
    return df


def load_legacy_weekly(path: str | Path) -> pd.DataFrame:
    """Carga y valida la fuente 2000-2008 (ya agregada por distrito x semana).

    El UBIGEO llega como entero en el CSV; se normaliza a string de 6 digitos
    con ceros a la izquierda, porque los codigos de departamentos de un solo
    digito (ej. Amazonas = 01) pierden el cero al leerse como numero.
    """
    df = pd.read_csv(path, dtype={"ubigeo": str})
    df["ubigeo"] = df["ubigeo"].str.zfill(6)
    df = _fill_missing_district_names(df)
    return LegacyWeeklySchema.validate(df)


def load_line_list(path: str | Path) -> pd.DataFrame:
    """Carga y valida la fuente 2009-2024 (line-list, un registro por caso)."""
    df = pd.read_csv(path, dtype={"ubigeo": str, "localcod": str}, low_memory=False)
    df["ubigeo"] = df["ubigeo"].str.zfill(6)
    out_of_range = (df["edad"] < MIN_EDAD) | (df["edad"] > MAX_EDAD)
    df.loc[out_of_range, "edad"] = float("nan")
    return LineListSchema.validate(df)
