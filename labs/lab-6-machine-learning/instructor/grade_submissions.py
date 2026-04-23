"""Lab 6 — calificación automática de entregas.

Lee todos los archivos `predictions_*.csv` de una carpeta, los une al
ground truth por `encounter_id`, calcula AUC por estudiante, y marca
entregas inválidas.

Uso:
    python instructor/grade_submissions.py --submissions <carpeta>

Salida: `grades.csv` en la carpeta de submissions.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
GROUND_TRUTH_PATH = HERE / "validation_ground_truth.csv"

VALID_FILENAME = re.compile(r"^predictions_([a-z0-9_\-]+)\.csv$", re.IGNORECASE)


def evaluate_submission(path: Path, gt: pd.DataFrame) -> dict:
    """Evalúa una entrega. Siempre retorna un dict con campos consistentes.

    Errores se marcan en 'status' y 'issue'; los válidos traen AUC.
    """
    base = {
        "file": path.name,
        "student": None,
        "status": "invalid",
        "issue": "",
        "auc": np.nan,
        "rows": 0,
    }

    m = VALID_FILENAME.match(path.name)
    if not m:
        base["issue"] = "nombre de archivo no coincide con predictions_<apellido>.csv"
        return base
    base["student"] = m.group(1).lower()

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        base["issue"] = f"no se pudo leer el CSV: {exc}"
        return base

    base["rows"] = len(df)

    # Columnas requeridas
    required = {"encounter_id", "prob_readmission"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        base["issue"] = f"faltan columnas: {sorted(missing_cols)}"
        return base

    # Cantidad de filas
    if len(df) != len(gt):
        base["issue"] = f"número de filas incorrecto: esperaba {len(gt)}, llegaron {len(df)}"
        return base

    # NaN / Inf
    proba = df["prob_readmission"]
    if proba.isna().any() or np.isinf(proba).any():
        base["issue"] = "hay NaN o Inf en prob_readmission"
        return base

    # Rango [0, 1]
    if (proba < 0).any() or (proba > 1).any():
        out_of_range = ((proba < 0) | (proba > 1)).sum()
        if out_of_range <= 5:
            df["prob_readmission"] = proba.clip(0, 1)
            base["issue"] = f"clipped {out_of_range} valores fuera de [0,1]"
        else:
            base["issue"] = f"{out_of_range} probabilidades fuera de [0,1]"
            return base

    # Merge con ground truth
    merged = gt.merge(df, on="encounter_id", how="left")
    if merged["prob_readmission"].isna().any():
        n_miss = merged["prob_readmission"].isna().sum()
        base["issue"] = f"{n_miss} encounter_id del ground truth no están en la entrega"
        return base

    # Calcular AUC
    try:
        auc = roc_auc_score(merged["target"], merged["prob_readmission"])
    except Exception as exc:
        base["issue"] = f"error calculando AUC: {exc}"
        return base

    base["status"] = "valid"
    base["auc"] = auc
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submissions", required=True, type=Path,
        help="Carpeta con archivos predictions_*.csv",
    )
    parser.add_argument(
        "--ground-truth", type=Path, default=GROUND_TRUTH_PATH,
        help="Ruta al archivo validation_ground_truth.csv",
    )
    args = parser.parse_args()

    if not args.ground_truth.exists():
        sys.exit(f"No encuentro el ground truth: {args.ground_truth}")
    if not args.submissions.is_dir():
        sys.exit(f"La carpeta no existe: {args.submissions}")

    gt = pd.read_csv(args.ground_truth)
    print(f"Ground truth: {len(gt)} filas, prevalencia {gt['target'].mean():.1%}")

    submissions = sorted(args.submissions.glob("predictions_*.csv"))
    print(f"Encontradas {len(submissions)} entregas\n")

    results = [evaluate_submission(p, gt) for p in submissions]
    if not results:
        sys.exit("Sin entregas para evaluar.")

    df = pd.DataFrame(results).sort_values(
        ["status", "auc"], ascending=[True, False]
    )

    out = args.submissions / "grades.csv"
    df.to_csv(out, index=False, encoding="utf-8")

    # Resumen
    n_valid = (df["status"] == "valid").sum()
    n_invalid = (df["status"] == "invalid").sum()
    print(f"Válidas: {n_valid}")
    print(f"Inválidas: {n_invalid}")
    if n_valid:
        valid_df = df[df["status"] == "valid"]
        print(f"\nAUC — mejor: {valid_df['auc'].max():.3f}")
        print(f"AUC — mediana: {valid_df['auc'].median():.3f}")
        print(f"AUC — peor: {valid_df['auc'].min():.3f}")
    print(f"\n{out.resolve()}")


if __name__ == "__main__":
    main()
