# Lab 6 — Machine Learning Supervisado

En los labs anteriores:

- Construiste tu primer modelo predictivo (regresión logística) sobre datos sintéticos de 108 pacientes (Lab 4)
- Aprendiste el flujo completo: feature engineering, CV, ROC/AUC, data leakage

Ahora la pregunta natural:

> ¿Qué otros algoritmos existen más allá de la regresión logística? ¿Cuándo conviene uno sobre otro? ¿Y cómo se comportan cuando el dataset es grande y real?

---

## Objetivo del laboratorio

Comparar **cuatro algoritmos de ML supervisado** (logística, árbol de decisión, Random Forest, SVM) sobre un dataset clínico real, y entregar un modelo de tu elección con la mejor performance posible sobre un *validation set* oculto.

---

## Qué vamos a construir

```
UCI Diabetes 130-US Hospitals ──┐
(1999-2008, 101,766 encuentros) │
                                ├──▶ Model zoo (6 modelos)  ──▶  Tu mejor modelo
  20,000 train + 2,000 val ─────┘
                                           │
                                           ▼
                              predictions_<apellido>.csv
```

**Cuatro algoritmos:**
1. **Regresión logística** (baseline heredado de Lab 4)
2. **Árbol de decisión** — nuevo
3. **Random Forest** — nuevo
4. **SVM lineal** — nuevo

**Problema:** predecir **readmisión hospitalaria a 30 días** a partir de datos del encuentro (demografía, diagnósticos, labs, medicamentos).

---

## Qué aprenderás

- Cómo entrenar y comparar árboles, Random Forest y SVM con scikit-learn
- Cómo leer feature importance e interpretarlo clínicamente
- Cómo funciona `predict_proba` de un Random Forest por dentro (no es voto por mayoría)
- Cómo decidir qué hiperparámetros valen la pena tunear
- Primera experiencia con una **competencia tipo Kaggle interna** contra un validation set oculto
- Escribir una **recomendación clínica** basada en tu modelo

---

## Flujo de trabajo

Sigue el flujo definido en `COURSE_SETUP.md`.

1. Actualiza `main`
2. Crea tu branch:

```bash
git checkout -b lab06-machine-learning/grupo-N
```

3. Trabaja únicamente en ese branch
4. Haz commits durante el proceso (hay 8 checkpoints en el notebook)
5. Entrega con Pull Request **+** el archivo `predictions_<tuapellido>.csv`

---

## Importante

Este laboratorio se guía completamente desde el notebook:

```
notebooks/Lab06_Machine_Learning_Supervisado.ipynb
```

Los datos ya vienen pre-generados en `data/`:
- `public_train.csv` — 20,000 encuentros con outcome
- `validation_features.csv` — 2,000 encuentros sin outcome (lo que tu modelo debe predecir)

---

## Antes de empezar

Verifica:

```bash
git branch
git remote -v
docker compose up -d
```

- No estás en `main`
- Tienes `origin` y `upstream`
- Docker está corriendo (jupyter disponible en http://localhost:8888)

---

## Resultado esperado

Al final deberías poder:

- Entrenar y comparar logística, árbol, Random Forest y SVM usando el mismo pipeline
- Entender qué hace cada algoritmo internamente (seguir un paciente por un árbol, verificar `predict_proba` de RF)
- Leer feature importance con ojo clínico
- Decidir qué modelo usar para un caso clínico concreto y justificarlo con números y contexto
- Entregar un archivo de predicciones válido sobre el validation set

---

## Para el instructor

La carpeta `instructor/` contiene:

- `prepare_dataset.py` — script para generar los CSVs desde UCI (correr una vez antes del curso)
- `validation_ground_truth.csv` — outcomes verdaderos del validation set (**no distribuir a estudiantes**)
- `grade_submissions.py` — script de grading automático (`python instructor/grade_submissions.py --submissions <carpeta_entregas>`)
