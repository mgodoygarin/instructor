"""
Lab 6 — Preparación del dataset UCI Diabetes 130-US Hospitals.

Este script corre UNA VEZ antes del curso para generar los archivos que se
distribuyen a los estudiantes. NO forma parte del notebook del lab.

Produce tres archivos:
  - data/public_train.csv         (con target, ~20k filas)
  - data/validation_features.csv  (sin target, 2k filas)
  - instructor/validation_ground_truth.csv  (con target, 2k filas — PRIVADO)

Claves del diseño:
  - Split por paciente (GroupShuffleSplit) para evitar que encuentros del
    mismo paciente aparezcan en train y validation.
  - encounter_id artificial ('enc_00001', ...) como columna explícita.
  - Codificación de diagnósticos a nivel de capítulo ICD-9 (primeros 3 chars).
  - Semilla fija 42 en todas las operaciones estocásticas.

Uso:
    pip install ucimlrepo pandas numpy scikit-learn
    python instructor/prepare_dataset.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from ucimlrepo import fetch_ucirepo

SEED = 42
TRAIN_TARGET_SIZE = 20_000
VALIDATION_TARGET_SIZE = 2_000

HERE = Path(__file__).resolve().parent
LAB_ROOT = HERE.parent
DATA_DIR = LAB_ROOT / "data"
INSTRUCTOR_DIR = LAB_ROOT / "instructor"

COLUMNS_TO_DROP = [
    "weight",
    "payer_code",
    "medical_specialty",
    "encounter_id",  # reemplazado por ID artificial
]

DIAG_COLUMNS = ["diag_1", "diag_2", "diag_3"]


def truncate_diag_code(code) -> str:
    """Convierte un código ICD-9 en su capítulo (primeros 3 caracteres).

    Valores faltantes o '?' se devuelven como 'unknown' para que el modelo
    los trate como una categoría más.
    """
    if pd.isna(code) or str(code).strip() in ("", "?"):
        return "unknown"
    return str(code).strip()[:3]


def binarize_target(readmitted: pd.Series) -> pd.Series:
    """<30 -> 1, resto (>30, NO) -> 0."""
    return (readmitted.astype(str).str.strip() == "<30").astype(int)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INSTRUCTOR_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Lab 6 — prepare_dataset.py")
    print("=" * 60)

    print("\n[1/6] Descargando UCI #296 via ucimlrepo ...")
    ds = fetch_ucirepo(id=296)
    ids = ds.data.ids.copy()          # encounter_id, patient_nbr
    features = ds.data.features.copy()
    targets = ds.data.targets.copy()  # readmitted
    df = ids.join(features).join(targets)
    print(f"    dataset original: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    print(f"    columnas de id: {list(ids.columns)}")

    print("\n[2/6] Binarizando outcome 'readmitted' ...")
    df["target"] = binarize_target(df["readmitted"])
    df = df.drop(columns=["readmitted"])
    prevalence = df["target"].mean()
    print(f"    prevalencia de readmisión <30d: {prevalence:.1%}")

    print("\n[3/6] Truncando diagnósticos a capítulos ICD-9 ...")
    for col in DIAG_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(truncate_diag_code)

    print("\n[4/6] Eliminando columnas problemáticas ...")
    # patient_nbr lo necesitamos para el split — se elimina DESPUÉS
    to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=to_drop)
    print(f"    eliminadas: {to_drop}")

    # Limpieza de '?' en columnas categóricas (los convertimos a NaN para que
    # el SimpleImputer del notebook los maneje consistentemente)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].replace("?", np.nan)

    print("\n[5/6] Split por paciente (GroupShuffleSplit) ...")
    n_patients = df["patient_nbr"].nunique()
    print(f"    {n_patients:,} pacientes únicos en {len(df):,} encuentros")

    # Primer split: validation vs rest por paciente
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
    rest_idx, val_idx = next(splitter.split(df, groups=df["patient_nbr"]))
    df_rest = df.iloc[rest_idx].reset_index(drop=True)
    df_val_full = df.iloc[val_idx].reset_index(drop=True)

    # Submuestreo dentro de cada lado para llegar a los tamaños objetivo,
    # manteniendo estratificación por target.
    def _sample_stratified(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
        if len(frame) <= n:
            return frame.sample(frac=1, random_state=seed).reset_index(drop=True)
        pos = frame[frame["target"] == 1]
        neg = frame[frame["target"] == 0]
        prev = pos.shape[0] / frame.shape[0]
        n_pos = int(round(n * prev))
        n_neg = n - n_pos
        return (
            pd.concat([
                pos.sample(n=min(n_pos, len(pos)), random_state=seed),
                neg.sample(n=min(n_neg, len(neg)), random_state=seed),
            ])
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )

    train_df = _sample_stratified(df_rest, TRAIN_TARGET_SIZE, SEED)
    val_df = _sample_stratified(df_val_full, VALIDATION_TARGET_SIZE, SEED + 1)

    # Verificación crítica: cero pacientes compartidos
    shared = set(train_df["patient_nbr"]) & set(val_df["patient_nbr"])
    assert not shared, f"LEAKAGE: {len(shared)} pacientes en ambos splits"
    print(f"    train: {len(train_df):,} encuentros, {train_df['patient_nbr'].nunique():,} pacientes")
    print(f"    val:   {len(val_df):,} encuentros, {val_df['patient_nbr'].nunique():,} pacientes")
    print(f"    ✓ sin pacientes compartidos entre splits")
    print(f"    prevalencia train: {train_df['target'].mean():.1%}")
    print(f"    prevalencia val:   {val_df['target'].mean():.1%}")

    print("\n[6/6] Asignando encounter_id artificial y guardando CSVs ...")
    # Drop patient_nbr ahora (ya no se necesita)
    train_df = train_df.drop(columns=["patient_nbr"])
    val_df = val_df.drop(columns=["patient_nbr"])

    # encounter_id artificial
    train_df.insert(0, "encounter_id",
                    [f"enc_t_{i:05d}" for i in range(1, len(train_df) + 1)])
    val_df.insert(0, "encounter_id",
                  [f"enc_v_{i:05d}" for i in range(1, len(val_df) + 1)])

    # public_train.csv — con target
    public_train_path = DATA_DIR / "public_train.csv"
    train_df.to_csv(public_train_path, index=False, encoding="utf-8")
    print(f"    ✓ {public_train_path.relative_to(LAB_ROOT)}  ({len(train_df):,} filas)")

    # validation_features.csv — sin target
    val_features = val_df.drop(columns=["target"])
    val_features_path = DATA_DIR / "validation_features.csv"
    val_features.to_csv(val_features_path, index=False, encoding="utf-8")
    print(f"    ✓ {val_features_path.relative_to(LAB_ROOT)}  ({len(val_features):,} filas, sin target)")

    # validation_ground_truth.csv — privado, para grading
    gt = val_df[["encounter_id", "target"]]
    gt_path = INSTRUCTOR_DIR / "validation_ground_truth.csv"
    gt.to_csv(gt_path, index=False, encoding="utf-8")
    print(f"    ✓ {gt_path.relative_to(LAB_ROOT)}  (PRIVADO — no distribuir)")

    print("\n" + "=" * 60)
    print("Listo. Resumen:")
    print("=" * 60)
    print(f"  public_train.csv         — {len(train_df):,} filas, {train_df.shape[1]} columnas")
    print(f"  validation_features.csv  — {len(val_features):,} filas, {val_features.shape[1]} columnas")
    print(f"  validation_ground_truth.csv — {len(gt):,} filas (privado)")
    print(f"\n  Columnas en public_train.csv:")
    for c in train_df.columns:
        print(f"    - {c} ({train_df[c].dtype})")


if __name__ == "__main__":
    main()
