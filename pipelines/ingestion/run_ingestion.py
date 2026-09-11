"""Orquesta la ingesta: carga las dos fuentes crudas, las une y valida,
y escribe la tabla canonica en data/gold/weekly_cases/.

Uso: ``python -m pipelines.ingestion.run_ingestion``
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.ingestion.renace import load_legacy_weekly, load_line_list
from src.preprocessing.transform import build_canonical_weekly_cases

CONFIG_PATH = Path("configs/data.yaml")


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    df_legacy = load_legacy_weekly(config["raw"]["legacy_weekly_csv"])
    df_line_list = load_line_list(config["raw"]["line_list_csv"])

    canonical = build_canonical_weekly_cases(df_legacy, df_line_list)

    output_path = Path(config["gold"]["canonical_weekly_cases"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_parquet(output_path, index=False)

    print(f"Ingesta completa: {len(canonical)} registros -> {output_path}")


if __name__ == "__main__":
    main()
