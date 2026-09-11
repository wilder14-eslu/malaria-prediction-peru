"""Esquemas de validacion (Pandera) para las dos fuentes crudas y la tabla canonica.

Hay dos fuentes de datos abiertos de vigilancia de malaria con formatos distintos:

- ``LegacyWeeklySchema``: fuente 2000-2008, ya agregada por distrito x semana
  epidemiologica, con conteo de casos por especie (falciparum/vivax).
- ``LineListSchema``: fuente 2009-2024, formato line-list (un registro por
  caso individual), con codigo de diagnostico CIE-10 (B50 = P. falciparum,
  B51 = P. vivax).

Ambas se unen en ``CanonicalWeeklyCasesSchema``, la tabla canonica que consume
el resto del pipeline (features, entrenamiento, inferencia).
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series

# Los UBIGEO peruanos son codigos de 6 digitos (departamento + provincia + distrito).
UBIGEO_REGEX = r"^\d{6}$"

MIN_EPI_YEAR = 1990
MAX_EPI_YEAR = 2100


class LegacyWeeklySchema(pa.DataFrameModel):
    """Fuente 2000-2008: ya agregada por distrito x semana epidemiologica."""

    ano: Series[int] = pa.Field(ge=MIN_EPI_YEAR, le=MAX_EPI_YEAR)
    semana: Series[int] = pa.Field(ge=1, le=53)
    departamento: Series[str]
    provincia: Series[str]
    distrito: Series[str]
    ubigeo: Series[str] = pa.Field(str_matches=UBIGEO_REGEX)
    falciparum: Series[int] = pa.Field(ge=0)
    vivax: Series[int] = pa.Field(ge=0)

    class Config:
        coerce = True
        strict = True


class LineListSchema(pa.DataFrameModel):
    """Fuente 2009-2024: un registro por caso individual (line-list)."""

    departamento: Series[str]
    provincia: Series[str]
    distrito: Series[str]
    localidad: Series[str] = pa.Field(nullable=True)
    enfermedad: Series[str]
    ano: Series[int] = pa.Field(ge=MIN_EPI_YEAR, le=MAX_EPI_YEAR)
    semana: Series[int] = pa.Field(ge=1, le=53)
    diagnostic: Series[str] = pa.Field(isin=["B50", "B51"])
    diresa: Series[str] = pa.Field(nullable=True)
    ubigeo: Series[str] = pa.Field(str_matches=UBIGEO_REGEX)
    localcod: Series[str] = pa.Field(nullable=True)
    # float, no int: la fuente real trae valores corruptos (ej. una fecha
    # pegada en la columna de edad) que se limpian a NaN en load_line_list()
    # antes de validar -- un entero de numpy no puede representar NaN.
    edad: Series[float] = pa.Field(ge=0, le=120, nullable=True)
    tipo_edad: Series[str] = pa.Field(nullable=True)
    sexo: Series[str] = pa.Field(isin=["M", "F"], nullable=True)

    class Config:
        coerce = True
        strict = False  # columnas extra ocasionales en exports oficiales no rompen la carga


class CanonicalWeeklyCasesSchema(pa.DataFrameModel):
    """Tabla canonica: un registro por UBIGEO x anio x semana epidemiologica."""

    ubigeo: Series[str] = pa.Field(str_matches=UBIGEO_REGEX)
    departamento: Series[str]
    provincia: Series[str]
    distrito: Series[str]
    epi_year: Series[int] = pa.Field(ge=MIN_EPI_YEAR, le=MAX_EPI_YEAR)
    epi_week: Series[int] = pa.Field(ge=1, le=53)
    cases_falciparum: Series[int] = pa.Field(ge=0)
    cases_vivax: Series[int] = pa.Field(ge=0)
    cases_total: Series[int] = pa.Field(ge=0)

    class Config:
        coerce = True
        strict = True
        unique = ["ubigeo", "epi_year", "epi_week"]
