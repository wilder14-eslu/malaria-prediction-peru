"""Features epidemiologicas derivadas de la composicion de especie de los casos.

Nota de diseno: el proyecto anterior tambien calculaba ``child_ratio`` y
``male_ratio`` (composicion demografica). Se omiten aqui a proposito: esos
datos (edad, sexo) solo existen en la fuente line-list (2009-2024); la fuente
legacy (2000-2008) no los tiene. Calcularlos dejaria esas columnas vacias en
~40% de la historia disponible, lo que complica mas de lo que aporta en esta
etapa basica. ``vivax_ratio``/``falciparum_ratio`` si estan disponibles en
toda la historia porque se derivan de ``cases_total``, presente en ambas
fuentes desde la ingesta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_species_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega ``vivax_ratio`` y ``falciparum_ratio`` (proporcion de casos por
    especie sobre el total de esa semana). NaN cuando ``cases_total`` es 0
    (sin casos, no hay proporcion que calcular), en vez de forzar un 0 que
    se confundiria con "0% de esa especie".
    """
    df = df.copy()
    has_cases = df["cases_total"] > 0
    df["vivax_ratio"] = np.where(has_cases, df["cases_vivax"] / df["cases_total"], np.nan)
    df["falciparum_ratio"] = np.where(has_cases, df["cases_falciparum"] / df["cases_total"], np.nan)
    return df
