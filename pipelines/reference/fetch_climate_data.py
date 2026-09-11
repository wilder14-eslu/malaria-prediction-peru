"""Descarga datos climaticos diarios (temperatura, precipitacion, humedad)
por departamento del Peru, desde la API publica y gratuita de NASA POWER
(https://power.larc.nasa.gov/, no requiere API key).

Por que a nivel DEPARTAMENTO y no distrito: no existe una fuente publica
gratuita de clima ya agregada a nivel distrital para el Peru, y consultar
~1800 puntos (uno por distrito) via API seria muy lento y probablemente
bloqueado por rate limiting. Usar 1 punto por departamento (25 en total,
el centroide de cada uno -- ver ``build_department_centroids.py``) es una
simplificacion real mencionada en ``docs/architecture.md``: pierde
variacion climatica dentro de un departamento grande como Loreto, pero es
la unica escala viable sin acceso a datos satelitales en bruto (que
requieren agregacion zonal por poligono sobre varios GB de NetCDF).

Este script SOLO usa la libreria estandar de Python (urllib, csv, json) a
proposito: se corre desde la maquina del usuario (necesita salida a
internet, que el entorno en la nube donde se construyo el resto del
pipeline no tiene hacia dominios fuera de un allowlist), y asi no hace
falta instalar nada antes de correrlo.

Reintenta con backoff exponencial ante errores de red/rate limiting, y es
reanudable: si se interrumpe, correrlo de nuevo salta los pares
departamento x anio que ya estan en el CSV de salida.

Uso: ``python pipelines/reference/fetch_climate_data.py``
(o con rango de anios: ``python pipelines/reference/fetch_climate_data.py
--start-year 2000 --end-year 2024``)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

CENTROIDS_PATH = Path("data/reference/department_centroids.csv")
OUTPUT_PATH = Path("data/raw/climate/climate_daily_by_department.csv")

API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
# T2M: temperatura media a 2m (C). PRECTOTCORR: precipitacion corregida
# (mm/dia). RH2M: humedad relativa a 2m (%). Las tres son las variables
# climaticas mas asociadas a la transmision de malaria en la literatura
# (ver "Fundamentacion" en docs/architecture.md).
PARAMETERS = "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M"
COMMUNITY = "AG"

MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 5
REQUEST_DELAY_SECONDS = 1.0  # cortesia con la API, evita rate limiting


def _load_centroids(path: Path = CENTROIDS_PATH) -> list[tuple[str, float, float]]:
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [(row["departamento"], float(row["lat"]), float(row["lon"])) for row in reader]


def _fetch_year(departamento: str, lat: float, lon: float, year: int) -> list[dict[str, str]]:
    url = (
        f"{API_URL}?parameters={PARAMETERS}&community={COMMUNITY}"
        f"&longitude={lon}&latitude={lat}"
        f"&start={year}0101&end={year}1231&format=JSON"
    )
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = json.loads(response.read())
            break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = exc
            wait = RETRY_BACKOFF_SECONDS * attempt
            print(
                f"  aviso: intento {attempt}/{MAX_RETRIES} fallo ({exc}), reintentando en {wait}s"
            )
            time.sleep(wait)
    else:
        raise RuntimeError(
            f"No se pudo descargar {departamento} {year} tras {MAX_RETRIES} intentos"
        ) from last_error

    params = payload["properties"]["parameter"]
    dates = sorted(params["T2M"].keys())
    rows = []
    for d in dates:
        rows.append(
            {
                "departamento": departamento,
                "date": f"{d[0:4]}-{d[4:6]}-{d[6:8]}",
                "t2m": params["T2M"][d],
                "t2m_max": params["T2M_MAX"][d],
                "t2m_min": params["T2M_MIN"][d],
                "precip_mm": params["PRECTOTCORR"][d],
                "rh2m": params["RH2M"][d],
            }
        )
    return rows


def _already_fetched(output_path: Path) -> set[tuple[str, int]]:
    if not output_path.exists():
        return set()
    done = set()
    with output_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((row["departamento"], int(row["date"][:4])))
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--centroids", type=Path, default=CENTROIDS_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    centroids = _load_centroids(args.centroids)
    done = _already_fetched(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    write_header = not args.output.exists()
    with args.output.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["departamento", "date", "t2m", "t2m_max", "t2m_min", "precip_mm", "rh2m"]
        )
        if write_header:
            writer.writeheader()

        total = len(centroids) * (args.end_year - args.start_year + 1)
        done_count = 0
        for departamento, lat, lon in centroids:
            for year in range(args.start_year, args.end_year + 1):
                done_count += 1
                if (departamento, year) in done:
                    print(f"[{done_count}/{total}] {departamento} {year}: ya descargado, se salta")
                    continue
                print(f"[{done_count}/{total}] {departamento} {year}: descargando...")
                rows = _fetch_year(departamento, lat, lon, year)
                writer.writerows(rows)
                f.flush()
                time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nListo. Clima diario por departamento escrito en {args.output}")


if __name__ == "__main__":
    main()
