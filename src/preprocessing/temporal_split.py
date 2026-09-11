"""Split train/val/test por anio (walk-forward), NUNCA aleatorio.

En series temporales, un split aleatorio deja filas del futuro en train y
filas del pasado en test, lo que infla artificialmente el desempeno medido
(el modelo "ve" el futuro indirectamente via filas cercanas en el tiempo).
El split walk-forward evita esto: todo train es estrictamente anterior a
todo val, que es estrictamente anterior a todo test.
"""

from __future__ import annotations

import pandas as pd


def walk_forward_split(
    df: pd.DataFrame,
    train_end_year: int,
    val_end_year: int,
    test_end_year: int,
    year_col: str = "epi_year",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Devuelve (train, val, test): train <= train_end_year < val <= val_end_year
    < test <= test_end_year.
    """
    train = df.loc[df[year_col] <= train_end_year]
    val = df.loc[(df[year_col] > train_end_year) & (df[year_col] <= val_end_year)]
    test = df.loc[(df[year_col] > val_end_year) & (df[year_col] <= test_end_year)]
    return train, val, test
