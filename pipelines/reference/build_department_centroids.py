"""Calcula el centroide de cada departamento del Peru, a partir de los mismos
limites distritales de INEI que usa ``build_district_adjacency.py``
(https://github.com/juaneladio/peru-geojson, licencia MPL-2.0).

Para que sirve: la Etapa 2 del rediseno ML (Temporal Fusion Transformer +
clima, ver ``docs/architecture.md``) necesita una serie climatica por
distrito x semana, pero no existe una fuente publica gratuita de clima ya
agregada a nivel distrital para el Peru. La aproximacion practica (y
documentada como tal) es usar UNA serie climatica por DEPARTAMENTO -- 25 en
vez de ~1800 puntos -- consultando la API de NASA POWER en el centroide de
cada departamento (``fetch_climate_data.py``). Es una simplificacion real
(pierde variacion climatica dentro de un departamento grande como Loreto),
pero es la unica escala viable sin acceso a datos satelitales en bruto
(varios GB, requieren agregacion zonal por poligono).

Requiere el extra ``geo`` (``pip install -e ".[geo]"``): calcula centroides
con shapely, igual que ``build_district_adjacency.py``.

Uso: ``python -m pipelines.reference.build_department_centroids``
"""

from __future__ import annotations

import csv
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

from shapely.geometry import shape
from shapely.ops import unary_union

GEOJSON_URL = (
    "https://raw.githubusercontent.com/juaneladio/peru-geojson/master/peru_distrital_simple.geojson"
)
CACHE_PATH = Path("data/reference/peru_distrital_simple.geojson")
OUTPUT_PATH = Path("data/reference/department_centroids.csv")


def _download_geojson(url: str = GEOJSON_URL, cache_path: Path = CACHE_PATH) -> Path:
    if cache_path.exists():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, cache_path)
    return cache_path


def compute_department_centroids(geojson_path: Path) -> list[tuple[str, float, float]]:
    """Devuelve filas (departamento, lat, lon) del centroide de la union de
    todos los distritos de cada departamento."""
    with geojson_path.open(encoding="utf-8") as f:
        data = json.load(f)

    dept_geoms: dict[str, list] = defaultdict(list)
    for feature in data["features"]:
        if feature.get("geometry") is None:
            continue
        departamento = feature["properties"]["NOMBDEP"]
        geom = shape(feature["geometry"])
        if not geom.is_valid:
            geom = geom.buffer(0)
        dept_geoms[departamento].append(geom)

    rows = []
    for departamento, geoms in sorted(dept_geoms.items()):
        centroid = unary_union(geoms).centroid
        rows.append((departamento, round(centroid.y, 4), round(centroid.x, 4)))
    return rows


def main() -> None:
    geojson_path = _download_geojson()
    rows = compute_department_centroids(geojson_path)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["departamento", "lat", "lon"])
        writer.writerows(rows)

    print(f"Centroides de departamento escritos: {len(rows)} filas -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
