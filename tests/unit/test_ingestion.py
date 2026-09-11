"""Tests unitarios del pipeline de ingesta: carga, agregacion y union de fuentes."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.renace import load_legacy_weekly, load_line_list
from src.preprocessing.transform import aggregate_line_list_to_weekly, build_canonical_weekly_cases
from src.validation.schemas import CanonicalWeeklyCasesSchema, LegacyWeeklySchema, LineListSchema

LEGACY_CSV = """ano,semana,departamento,provincia,distrito,ubigeo,falciparum,vivax
2008,1,LORETO,MAYNAS,IQUITOS,160101,5,10
2008,2,LORETO,MAYNAS,IQUITOS,160101,3,8
"""

LINE_LIST_CSV = """departamento,provincia,distrito,localidad,enfermedad,ano,semana,diagnostic,diresa,ubigeo,localcod,edad,tipo_edad,sexo
LORETO,MAYNAS,IQUITOS,IQUITOS,MALARIA POR P. VIVAX,2024,1,B51,LORETO,160101,001,25,A,M
LORETO,MAYNAS,IQUITOS,IQUITOS,MALARIA P. FALCIPARUM,2024,1,B50,LORETO,160101,001,30,A,F
LORETO,MAYNAS,IQUITOS,IQUITOS,MALARIA POR P. VIVAX,2024,1,B51,LORETO,160101,001,40,A,M
"""


@pytest.fixture
def df_legacy() -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(LEGACY_CSV), dtype={"ubigeo": str})
    return LegacyWeeklySchema.validate(df)


@pytest.fixture
def df_line_list() -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(LINE_LIST_CSV), dtype={"ubigeo": str, "localcod": str})
    return LineListSchema.validate(df)


def test_aggregate_line_list_to_weekly_counts_by_species(df_line_list: pd.DataFrame) -> None:
    result = aggregate_line_list_to_weekly(df_line_list)

    row = result.loc[
        (result["ubigeo"] == "160101") & (result["epi_year"] == 2024) & (result["epi_week"] == 1)
    ].iloc[0]

    assert row["vivax"] == 2
    assert row["falciparum"] == 1


def test_build_canonical_weekly_cases_unions_both_sources(
    df_legacy: pd.DataFrame, df_line_list: pd.DataFrame
) -> None:
    canonical = build_canonical_weekly_cases(df_legacy, df_line_list)

    CanonicalWeeklyCasesSchema.validate(canonical)
    assert len(canonical) == 3  # 2 semanas legacy (2008) + 1 semana line-list (2024)

    row_2024 = canonical.loc[canonical["epi_year"] == 2024].iloc[0]
    assert row_2024["cases_total"] == 3
    assert row_2024["cases_falciparum"] == 1
    assert row_2024["cases_vivax"] == 2


def test_build_canonical_weekly_cases_sums_when_sources_overlap(
    df_legacy: pd.DataFrame, df_line_list: pd.DataFrame
) -> None:
    # Simula un solape: el mismo UBIGEO x anio x semana aparece en ambas fuentes.
    df_line_list_overlap = df_line_list.copy()
    df_line_list_overlap["ano"] = 2008
    df_line_list_overlap["semana"] = 1

    canonical = build_canonical_weekly_cases(df_legacy, df_line_list_overlap)
    row = canonical.loc[(canonical["epi_year"] == 2008) & (canonical["epi_week"] == 1)].iloc[0]

    # Legacy: falciparum=5, vivax=10. Line-list solapado: falciparum=1, vivax=2.
    assert row["cases_falciparum"] == 6
    assert row["cases_vivax"] == 12


def test_load_legacy_weekly_fills_missing_names_from_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Reproduce el caso real: UBIGEO valido pero departamento/provincia/
    # distrito vacios en la fuente (confirmado en el dataset real del
    # portal, UBIGEO 160109/160114).
    raw_csv = tmp_path / "legacy.csv"
    raw_csv.write_text(
        "ano,semana,departamento,provincia,distrito,ubigeo,falciparum,vivax\n2005,3,,,,160109,2,1\n"
    )
    reference_csv = tmp_path / "district_names.csv"
    reference_csv.write_text(
        "ubigeo,departamento,provincia,distrito\n160109,LORETO,MAYNAS,PUTUMAYO\n"
    )
    monkeypatch.setattr("src.ingestion.renace.DISTRICT_NAMES_PATH", reference_csv)

    result = load_legacy_weekly(raw_csv)

    row = result.iloc[0]
    assert row["departamento"] == "LORETO"
    assert row["provincia"] == "MAYNAS"
    assert row["distrito"] == "PUTUMAYO"


def test_load_line_list_nulls_out_of_range_edad(tmp_path: Path) -> None:
    # Reproduce el caso real: valores de edad corruptos (ej. una fecha
    # pegada en la columna) que no deberian tumbar la carga del resto de
    # filas validas.
    raw_csv = tmp_path / "line_list.csv"
    raw_csv.write_text(
        "departamento,provincia,distrito,localidad,enfermedad,ano,semana,diagnostic,"
        "diresa,ubigeo,localcod,edad,tipo_edad,sexo\n"
        "LORETO,MAYNAS,IQUITOS,IQUITOS,MALARIA POR P. VIVAX,2024,1,B51,LORETO,160101,001,"
        "90130297,A,M\n"
        "LORETO,MAYNAS,IQUITOS,IQUITOS,MALARIA POR P. VIVAX,2024,1,B51,LORETO,160101,001,"
        "25,A,F\n"
    )

    result = load_line_list(raw_csv)

    assert pd.isna(result.iloc[0]["edad"])
    assert result.iloc[1]["edad"] == 25
