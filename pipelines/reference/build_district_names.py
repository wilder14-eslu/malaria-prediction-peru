"""Construye una tabla de referencia UBIGEO -> nombres (departamento,
provincia, distrito) a partir de los limites distritales de INEI publicados
en https://github.com/juaneladio/peru-geojson (licencia MPL-2.0).

Por que hace falta: los datos abiertos de vigilancia (fuente 2000-2008)
traen, para un puñado de registros, el UBIGEO correcto pero los campos de
nombre (departamento/provincia/distrito) vacios -- se confirmo con los
datos reales del portal (473 de 106355 filas, los 2 UBIGEO 160109 y 160114,
ambos en Loreto/Maynas). Esta tabla permite rellenar esos nombres en vez de
descartar casos reales de malaria solo porque el nombre viene vacio.

A diferencia de ``build_district_adjacency.py``, este script NO necesita
shapely (no calcula geometria, solo lee las propiedades de cada feature),
asi que no requiere el extra "geo".

Uso: ``python -m pipelines.reference.build_district_names``
"""

from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

GEOJSON_URL = (
    "https://raw.githubusercontent.com/juaneladio/peru-geojson/master/peru_distrital_simple.geojson"
)
CACHE_PATH = Path("data/reference/peru_distrital_simple.geojson")
OUTPUT_PATH = Path("data/reference/district_names.csv")


def _download_geojson(url: str = GEOJSON_URL, cache_path: Path = CACHE_PATH) -> Path:
    if cache_path.exists():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, cache_path)
    return cache_path


def extract_district_names(geojson_path: Path) -> list[tuple[str, str, str, str]]:
    """Devuelve filas (ubigeo, departamento, provincia, distrito), una por distrito."""
    with geojson_path.open(encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for feature in data["features"]:
        props = feature["properties"]
        rows.append((props["IDDIST"], props["NOMBDEP"], props["NOMBPROV"], props["NOMBDIST"]))
    return sorted(rows)


def main() -> None:
    geojson_path = _download_geojson()
    rows = extract_district_names(geojson_path)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ubigeo", "departamento", "provincia", "distrito"])
        writer.writerows(rows)

    print(f"Nombres de distrito escritos: {len(rows)} filas -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
