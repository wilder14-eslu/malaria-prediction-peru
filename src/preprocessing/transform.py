"""Transformacion y union de las dos fuentes crudas a la tabla canonica.

``CanonicalWeeklyCasesSchema`` es el contrato que consume el resto del
pipeline (features, entrenamiento, inferencia): un registro por
UBIGEO x anio epidemiologico x semana epidemiologica.
"""

from __future__ import annotations

import pandas as pd

from src.validation.schemas import CanonicalWeeklyCasesSchema

DIAGNOSTIC_SPECIES_MAP = {"B50": "falciparum", "B51": "vivax"}


def aggregate_line_list_to_weekly(df_line_list: pd.DataFrame) -> pd.DataFrame:
    """Agrega el line-list (un registro por caso) a nivel UBIGEO x semana x especie.

    La especie se deriva del codigo CIE-10 (``diagnostic``), no del texto libre
    de ``enfermedad``, porque el codigo es el campo estandarizado y estable.
    """
    df = df_line_list.copy()
    df["species"] = df["diagnostic"].map(DIAGNOSTIC_SPECIES_MAP)

    grouped = (
        df.groupby(["ubigeo", "departamento", "provincia", "distrito", "ano", "semana", "species"])
        .size()
        .rename("cases")
        .reset_index()
    )
    wide = grouped.pivot_table(
        index=["ubigeo", "departamento", "provincia", "distrito", "ano", "semana"],
        columns="species",
        values="cases",
        fill_value=0,
    ).reset_index()
    wide.columns.name = None

    for species in DIAGNOSTIC_SPECIES_MAP.values():
        if species not in wide.columns:
            wide[species] = 0

    return wide.rename(columns={"ano": "epi_year", "semana": "epi_week"})


def build_canonical_weekly_cases(
    df_legacy: pd.DataFrame, df_line_list: pd.DataFrame
) -> pd.DataFrame:
    """Une la fuente legacy (2000-2008) con el line-list agregado (2009-2024).

    Los dos periodos no se solapan en la practica (la fuente legacy llega
    hasta 2008 y el line-list arranca en 2009), pero si algun UBIGEO x semana
    aparece en ambas se suman los casos en vez de descartar uno, para no
    perder informacion silenciosamente.
    """
    legacy = df_legacy.rename(
        columns={
            "ano": "epi_year",
            "semana": "epi_week",
            "falciparum": "cases_falciparum",
            "vivax": "cases_vivax",
        }
    )

    line_list_agg = aggregate_line_list_to_weekly(df_line_list).rename(
        columns={"falciparum": "cases_falciparum", "vivax": "cases_vivax"}
    )

    combined = pd.concat([legacy, line_list_agg], ignore_index=True)

    keys = ["ubigeo", "departamento", "provincia", "distrito", "epi_year", "epi_week"]
    canonical = (
        combined.groupby(keys, as_index=False)[["cases_falciparum", "cases_vivax"]]
        .sum()
        .astype({"cases_falciparum": int, "cases_vivax": int})
    )
    canonical["cases_total"] = canonical["cases_falciparum"] + canonical["cases_vivax"]

    return CanonicalWeeklyCasesSchema.validate(canonical)
