"""Construye la tabla de adyacencia entre distritos de Peru (que UBIGEO es
vecino de cual), a partir de los limites distritales de INEI publicados en
https://github.com/juaneladio/peru-geojson (licencia MPL-2.0).

Es un paso de referencia (se corre una vez, no en cada entrenamiento): baja
el GeoJSON nacional, calcula que poligonos se tocan usando un indice
espacial (STRtree de shapely) y escribe ``data/reference/district_adjacency.csv``.

Requiere el extra ``geo`` (``pip install -e ".[geo]"``), porque shapely solo
hace falta aqui, no en el resto del pipeline de features/entrenamiento.

Limitaciones conocidas (documentadas para no asumir mas precision de la
que hay):
- Los limites son de INEI ~2007: 1826 distritos con geometria valida, de
  los ~1874 que existen actualmente en Peru (los creados despues no estan).
  8 distritos del archivo fuente no tienen geometria y se excluyen.
- La adyacencia es "los poligonos se tocan o casi se tocan" (con una
  tolerancia de ~50m para compensar bordes que no calzan exacto por
  digitalizacion), no "estan conectados por una via transitable" ni
  considera movilidad real de personas entre distritos.

Uso: ``python -m pipelines.reference.build_district_adjacency``
"""

from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

from shapely.geometry import shape
from shapely.strtree import STRtree

GEOJSON_URL = (
    "https://raw.githubusercontent.com/juaneladio/peru-geojson/master/peru_distrital_simple.geojson"
)
CACHE_PATH = Path("data/reference/peru_distrital_simple.geojson")
OUTPUT_PATH = Path("data/reference/district_adjacency.csv")

# Tolerancia (~50m en grados decimales) para que bordes que no calzan exacto
# por digitalizacion sigan detectandose como vecinos.
ADJACENCY_TOLERANCE_DEG = 0.0005


def _download_geojson(url: str = GEOJSON_URL, cache_path: Path = CACHE_PATH) -> Path:
    if cache_path.exists():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, cache_path)
    return cache_path


def compute_adjacency_pairs(
    geojson_path: Path, tolerance_deg: float = ADJACENCY_TOLERANCE_DEG
) -> list[tuple[str, str]]:
    """Devuelve pares (ubigeo, neighbor_ubigeo) dirigidos (ambos sentidos)."""
    with geojson_path.open(encoding="utf-8") as f:
        data = json.load(f)

    ubigeos: list[str] = []
    geoms = []
    skipped = 0
    for feature in data["features"]:
        ubigeo = feature["properties"]["IDDIST"]
        if feature.get("geometry") is None:
            skipped += 1
            continue
        geom = shape(feature["geometry"])
        if not geom.is_valid:
            geom = geom.buffer(0)
        ubigeos.append(ubigeo)
        geoms.append(geom)

    if skipped:
        print(f"Aviso: {skipped} distritos sin geometria en la fuente, excluidos.")

    buffered = [g.buffer(tolerance_deg) for g in geoms]
    tree = STRtree(buffered)

    pairs: set[tuple[str, str]] = set()
    for i, g in enumerate(buffered):
        for j in tree.query(g):
            j = int(j)
            if j == i:
                continue
            if geoms[i].intersects(buffered[j]):
                a, b = ubigeos[i], ubigeos[j]
                if a != b:
                    pairs.add((a, b))

    return sorted(pairs)


def main() -> None:
    geojson_path = _download_geojson()
    pairs = compute_adjacency_pairs(geojson_path)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ubigeo", "neighbor_ubigeo"])
        writer.writerows(pairs)

    n_districts = len({a for a, _ in pairs})
    print(f"Adyacencia calculada: {len(pairs)} pares, {n_districts} distritos -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
