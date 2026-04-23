# Lab 6 — Machine Learning Supervisado: Comparación de Algoritmos

**Curso:** BE3006 · Análisis de Datos Biomédicos · UVG
**Profesor:** Miguel Godoy Garin
**Referencia teórica:** Clase 11 — Machine Learning I (supervisado clásico y clasificación)
**Lab previo:** Lab 4 — Modelado Estadístico

Este documento es la especificación para construir el notebook del Lab 6. El agente implementador debe seguirla al pie de la letra.

---

## 1. Filosofía del lab

Es el **primer lab** donde los estudiantes construyen modelos por cuenta propia sin seguir instrucciones paso a paso. Por eso:

- **Simplificar todo lo simplificable.** Si una decisión técnica puede tomarla el notebook en vez del estudiante, que la tome el notebook.
- **El estudiante toma dos decisiones:** (1) qué modelo usar, (2) cómo ajustar hiperparámetros. Todo lo demás — preprocessing, splits, scoring — viene preparado.
- **Guiar con explicaciones, no con fricción.** Si algo puede salir mal (SVM degenerado, árbol ilegible, preprocessing roto), que no salga mal. No es el lugar para aprender a debuggear sklearn.
- **Errores pedagógicos sí, errores de plomería no.** El overfitting de `max_depth=None` es un error que queremos mostrar. El `ConvergenceWarning` de logística no es.

---

## 2. Contexto del curso

Este lab sigue al Lab 4 y complementa la Clase 11.

**Lo que ya saben (Lab 4):**
- Feature engineering, EDA clínico, pipelines de sklearn
- Regresión logística completa: train/test split, CV estratificada, ROC/AUC, matriz de confusión, selección de umbral, calibración
- Data leakage, regularización L1/L2

**Lo que aporta este lab:**
- Árboles de decisión, Random Forest, SVM como nuevos algoritmos
- Comparación crítica entre algoritmos como proceso de decisión
- Primera experiencia con un dataset clínico real y grande (no Synthea)
- **Entrega con modelo contra un validation set que no han visto**
- Recomendación clínica escrita basada en el modelo

---

## 3. Estructura general

- **Duración:** 2 sesiones (~4 horas) o 1 sesión + tarea en casa
- **Modalidad:** primera mitad guiada, segunda mitad semi-libre (elegir modelo + hiperparámetros)
- **Dataset:** UCI Diabetes 130-US Hospitals (ver sección 4)
- **Entrega:** notebook + `predictions_<apellido>.csv` con probabilidades sobre el validation set

---

## 4. Dataset

### 4.1 Fuente

**UCI ML Repository #296** — "Diabetes 130-US Hospitals for Years 1999-2008"
- Licencia: CC BY 4.0
- Tamaño original: 101,766 encuentros × 50 features, ~71,500 pacientes únicos
- Outcome: `readmitted` binarizado a "readmisión <30 días"

### 4.2 Script `prepare_dataset.py`

El script corre una vez antes del curso y produce los CSVs que se distribuyen a los estudiantes. Pasos:

1. **Descargar con `ucimlrepo`:**
   ```python
   from ucimlrepo import fetch_ucirepo
   ds = fetch_ucirepo(id=296)
   df = ds.data.features.join(ds.data.targets)
   ```

2. **Binarizar outcome:** `readmitted = '<30'` → 1, resto → 0. Renombrar a `target`.

3. **Eliminar columnas problemáticas:**
   - `weight` (97% missing)
   - `payer_code` (49% missing, no clínico)
   - `medical_specialty` (49% missing)
   - `encounter_id` (se reemplaza por ID artificial abajo)

4. **Truncar diagnósticos:** `diag_1`, `diag_2`, `diag_3` a primeros 3 caracteres (capítulo ICD-9).

5. **CRÍTICO — split por paciente, no por encuentro:**
   - Usar `GroupShuffleSplit` con `groups=df['patient_nbr']`
   - Primero separar validation (10% de pacientes) del resto
   - Luego submuestrear el resto si excede el objetivo
   - Esto garantiza que ningún paciente aparece en train y validation
   - Tras el split, eliminar `patient_nbr` de los CSVs públicos

6. **Tamaño objetivo:** 20,000 encuentros en train público, 2,000 en validation.
   - Justificación del validation más grande que en la spec original: con 2,000 filas y prevalencia ~11% (~220 positivos), el IC 95% de AUC ~0.65 es aprox. ±0.02 en vez de ±0.03 con 1,000. Grading más justo.

7. **Asignar `encounter_id` artificial** (`enc_00001`, `enc_00002`, ...) como columna explícita en los CSVs públicos. Esto permite matchear predicciones al ground truth sin depender del orden de filas.

8. **Archivos producidos:**
   - `public_train.csv` — ~20,000 filas, **con** columna `target`, con `encounter_id` artificial
   - `validation_features.csv` — 2,000 filas, **SIN** columna `target`, con `encounter_id`
   - `validation_ground_truth.csv` — 2,000 filas con `encounter_id` y `target` (solo para grading, NO se publica)

9. **Semilla fija** (`random_state=42`).

10. **Output del script:** imprimir shape, prevalencia del outcome en cada split, número de pacientes únicos por split, y confirmar que no hay `patient_nbr` compartidos entre train y validation.

### 4.3 Benchmarks esperados (con split por paciente)

El split por paciente baja el AUC observado en ~0.02 respecto a un random split, porque elimina la memorización de pacientes recurrentes. Valores esperados:

| Algoritmo | Tiempo en 20k train | AUC esperado |
|---|---|---|
| Logística (defaults sensatos) | ~15s | 0.62–0.64 |
| Árbol (`max_depth=5`) | ~2s | 0.60–0.63 |
| Random Forest (200) | ~10s | 0.62–0.64 |
| SVM lineal | ~30s | 0.60–0.63 |

Techo práctico con estos modelos + feature engineering razonable: **AUC ≈ 0.65**.

---

## 5. Estructura del notebook

7 partes con checkpoints de Git entre cada una, siguiendo Lab 4.

### Parte 0: Setup y carga (guiado — ~10 min)

Código pre-llenado al 100%. Sin ejercicios.

- Importar librerías
- Cargar `public_train.csv` y `validation_features.csv`
- Mostrar `.shape`, `.head()`, `.info()`
- Explicación clínica breve: readmisión a 30 días, HRRP Medicare, costos, calidad

### Parte 1: EDA orientada al modelo (guiado + 1 ejercicio — ~15 min)

Pre-llenado al 80%. Celdas listas para correr:
- Distribución del outcome (balance)
- Missing values por columna
- Tipo de columna (numérica vs categórica)
- Correlación de numéricas con target

**Ejercicio 1.1 (pregunta abierta):**
> El outcome está desbalanceado (~11% positivos). En 2–3 oraciones: ¿por qué la **accuracy** será engañosa en este lab? ¿Por qué AUC es más informativa?

### Parte 2: Preprocessing compartido (guiado — ~10 min)

**Pre-llenado al 100%.** Esta es la pieza que evita frustración. El notebook construye **una sola función** que todos los modelos van a usar:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def build_pipeline(model):
    """
    Construye un pipeline de sklearn con preprocessing estándar
    (imputación + escalado + OHE) seguido del modelo que pases.
    Usa esta función para TODOS los modelos del lab.
    """
    num_features = X.select_dtypes(include=['number']).columns.tolist()
    cat_features = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', max_categories=20))
        ]), cat_features)
    ])

    return Pipeline([('prep', preprocessor), ('model', model)])
```

Y una función de evaluación:

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

def evaluate(model, X, y, cv=5):
    """
    Entrena con CV estratificada y reporta AUC promedio ± desviación.
    Úsala en TODOS tus experimentos.
    """
    pipe = build_pipeline(model)
    scores = cross_val_score(pipe, X, y, cv=StratifiedKFold(cv, shuffle=True, random_state=42),
                             scoring='roc_auc', n_jobs=-1)
    print(f"AUC: {scores.mean():.3f} ± {scores.std():.3f}")
    return scores
```

**Esto es la clave del lab.** El estudiante nunca tiene que volver a pensar en preprocessing. Solo pasa modelos a `evaluate(...)` y compara.

### Parte 3: Baseline — regresión logística (guiado + 1 ejercicio — ~15 min)

Pre-llenado al 90%. Recap corto de Lab 4.

- Una llamada: `evaluate(LogisticRegression(max_iter=1000, random_state=42), X, y)`
- Mostrar matriz de confusión sobre un holdout 80/20 del train público
- ROC curve

**Ejercicio 3.1:**
> Corre `evaluate` con la logística base. Reporta el AUC. Este es tu **piso** — cualquier modelo más complejo tiene que superarlo para justificar su costo. Anótalo para referencia.

### Parte 4: Árbol de decisión (guiado + 2 ejercicios — ~25 min)

**Dos árboles con propósitos distintos:**

#### 4a. Árbol visual (para entender el concepto)

Entrenar un árbol sobre **solo 5 features numéricas** (`age`, `time_in_hospital`, `num_medications`, `num_lab_procedures`, `number_diagnoses`) **sin pipeline**, para que `plot_tree` sea legible. Esto evita el problema de features OHE con nombres ilegibles.

```python
viz_features = ['time_in_hospital', 'num_medications',
                'num_lab_procedures', 'number_diagnoses', 'num_procedures']
X_viz = X[viz_features].fillna(X[viz_features].median())
tree_viz = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_viz, y)
plot_tree(tree_viz, feature_names=viz_features, filled=True, rounded=True)
```

**Ejercicio 4.1 (follow the patient):**
> Elige un paciente del test set (cualquiera). Imprime sus 5 features. Sigue manualmente la ruta del árbol (`X[feature] <= threshold?` → izquierda/derecha) hasta llegar a una hoja. Compara la probabilidad de la hoja con `tree_viz.predict_proba(paciente)[:,1]`.

Nota al estudiante: este árbol es "de juguete" para entender el concepto. El árbol real (para comparar AUC) usa todas las features vía `build_pipeline`.

#### 4b. Árbol real (para comparar performance)

```python
evaluate(DecisionTreeClassifier(max_depth=5, random_state=42), X, y)
```

**Ejercicio 4.2 (bias-variance):**
> Usa `evaluate` con tres árboles: `max_depth` en [3, 8, None]. Reporta el AUC de cada uno. ¿Cuál overfittea? ¿Cuál está subajustado? Conecta con bias-variance de Clase 9 en 2–3 oraciones.

### Parte 5: Random Forest (guiado + 2 ejercicios — ~25 min)

Pre-llenado al 70%.

```python
evaluate(RandomForestClassifier(n_estimators=200, max_depth=10,
                                 random_state=42, n_jobs=-1), X, y)
```

**Sobre `predict_proba` de RF — explicación correcta:**

El notebook debe explicar que `RandomForestClassifier.predict_proba(x)` devuelve el **promedio** de `tree.predict_proba(x)` sobre todos los árboles, **no** una votación por mayoría. Para `max_depth` pequeño con hojas puras coincide con voto mayoritario, pero en general no.

**Ejercicio 5.1 (verificar `predict_proba` manualmente):**

Para que esto funcione correctamente con pipeline, el código del ejercicio viene pre-escrito:

```python
# Fit pipeline RF sobre X train completo
rf_pipe = build_pipeline(RandomForestClassifier(n_estimators=100,
                                                  max_depth=5,
                                                  random_state=42))
rf_pipe.fit(X_train, y_train)

# Elige 1 paciente de test
paciente = X_test.iloc[[0]]

# Preprocess el paciente (extraer del pipeline)
prep = rf_pipe.named_steps['prep']
rf = rf_pipe.named_steps['model']
paciente_prep = prep.transform(paciente)

# Saca predict_proba de cada árbol y promedia manualmente
probas_por_arbol = np.array([tree.predict_proba(paciente_prep)[0, 1]
                              for tree in rf.estimators_])

print(f"Media manual de árboles: {probas_por_arbol.mean():.3f}")
print(f"predict_proba del forest: {rf_pipe.predict_proba(paciente)[0, 1]:.3f}")
print(f"¿Coinciden? {np.isclose(probas_por_arbol.mean(), rf_pipe.predict_proba(paciente)[0, 1])}")
```

El estudiante solo tiene que **correrlo y explicar** qué observa en 1–2 oraciones.

#### Visualizaciones del Random Forest (pre-llenadas, ~5 min)

Tres gráficas que hacen tangible el RF:

**(1) Un árbol individual del forest** — muestra que RF = muchos árboles distintos.

```python
# Truncado a 3 niveles para legibilidad
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(rf.estimators_[0], max_depth=3, filled=True, rounded=True,
          fontsize=9, ax=ax)
ax.set_title("Un árbol cualquiera del forest (truncado a 3 niveles)")
plt.show()
```

**(2) Distribución de predicciones entre los 100 árboles para un paciente** — hace visible la incertidumbre interna del modelo.

```python
plt.figure(figsize=(10, 5))
plt.hist(probas_por_arbol, bins=20, color='steelblue',
         edgecolor='white', alpha=0.8)
plt.axvline(probas_por_arbol.mean(), color='coral', linestyle='--',
            linewidth=2, label=f'Media = predict_proba = {probas_por_arbol.mean():.3f}')
plt.xlabel('Probabilidad de readmisión predicha por cada árbol')
plt.ylabel('Número de árboles')
plt.title('¿En qué se ponen de acuerdo los 100 árboles? (paciente de ejemplo)')
plt.legend()
plt.show()
```

**(3) Feature importance — top 15** — qué variables está usando el RF para decidir.

```python
# Nombres de features post-preprocessing
feature_names = rf_pipe.named_steps['prep'].get_feature_names_out()
importances = pd.Series(rf.feature_importances_, index=feature_names)
top15 = importances.sort_values(ascending=True).tail(15)

plt.figure(figsize=(10, 7))
top15.plot(kind='barh', color='steelblue')
plt.xlabel('Importancia (reducción de impureza promedio)')
plt.title('Top 15 features más importantes para el Random Forest')
plt.tight_layout()
plt.show()
```

**Ejercicio 5.2 (`n_estimators`):**
> Usa `evaluate` con `n_estimators` en [10, 50, 200, 500]. Reporta los AUC. ¿Mejora indefinidamente con más árboles? ¿Dónde se estanca?

**Pregunta 5.3 (lectura del feature importance):**
> Mira las 3 features más importantes según el RF. ¿Tienen sentido clínico para predecir readmisión a 30 días? Si alguna te sorprende, especula por qué aparece arriba.

### Parte 6: SVM (guiado + 1 ejercicio — ~10 min)

**Sección deliberadamente corta.** SVM con RBF en 20k filas es demasiado lento y frustrante. Usamos solo SVM lineal, vía `LinearSVC` envuelto en `CalibratedClassifierCV` para tener `predict_proba`.

```python
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

svm = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000,
                                        class_weight='balanced',
                                        random_state=42),
                              cv=3)
evaluate(svm, X, y)
```

Explicación: "SVM es más lento que logística y rara vez supera a RF en datos tabulares. Lo vemos para que lo conozcas; si en tus experimentos de la Parte 7 no compite, descártalo."

**Ejercicio 6.1:**
> Prueba tres valores de `C` en [0.01, 1, 100]. Reporta AUC. ¿Qué hace `C` intuitivamente? (una oración)

### Parte 7: Tu mejor modelo (semi-libre — ~45 min)

Esta es la pieza donde el estudiante toma decisiones. Para evitar el efecto "no sé por dónde empezar", el notebook provee **un model zoo** pre-escrito.

#### 7.1 Model zoo

```python
from sklearn.ensemble import VotingClassifier

candidates = {
    'logistica_L2': LogisticRegression(C=1.0, penalty='l2',
                                         max_iter=1000,
                                         class_weight='balanced',
                                         random_state=42),

    'logistica_L1': LogisticRegression(C=0.5, penalty='l1',
                                         solver='saga',
                                         max_iter=2000,
                                         class_weight='balanced',
                                         random_state=42),

    'arbol_profundo': DecisionTreeClassifier(max_depth=8,
                                               class_weight='balanced',
                                               random_state=42),

    'random_forest_pequeno': RandomForestClassifier(n_estimators=100,
                                                      max_depth=8,
                                                      class_weight='balanced',
                                                      random_state=42,
                                                      n_jobs=-1),

    'random_forest_grande': RandomForestClassifier(n_estimators=500,
                                                     max_depth=15,
                                                     class_weight='balanced',
                                                     random_state=42,
                                                     n_jobs=-1),

    'svm_lineal': CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000,
                                                      class_weight='balanced',
                                                      random_state=42),
                                           cv=3),
}

# Evalúa todos, guardando AUC promedio y desviación
resultados = {}
for nombre, modelo in candidates.items():
    print(f"\n=== {nombre} ===")
    scores = evaluate(modelo, X, y)
    resultados[nombre] = {'auc_mean': scores.mean(), 'auc_std': scores.std()}

df_resultados = pd.DataFrame(resultados).T.sort_values('auc_mean', ascending=False)
df_resultados
```

#### 7.2 Visualizaciones comparativas (pre-llenadas)

**Bar chart con error bars — quién ganó y si las diferencias son significativas:**

```python
df_plot = df_resultados.sort_values('auc_mean')
plt.figure(figsize=(10, 6))
plt.barh(df_plot.index, df_plot['auc_mean'], xerr=df_plot['auc_std'],
         color='steelblue', alpha=0.85, edgecolor='white',
         error_kw={'ecolor': 'coral', 'capsize': 4, 'capthick': 1.5})
plt.xlabel('AUC (CV 5-fold, media ± std)')
plt.title('Comparación de modelos del zoo')
plt.axvline(0.5, color='gray', linestyle=':', alpha=0.5, label='AUC aleatorio')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
```

Nota al estudiante: si los error bars de dos modelos se superponen, la diferencia puede estar dentro del ruido de CV. Ese es un test visual rápido de "¿es realmente mejor?".

**ROC curves superpuestas de los top 3 modelos:**

```python
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                           stratify=y, random_state=42)

top3 = df_resultados.head(3).index.tolist()
plt.figure(figsize=(8, 8))

for nombre in top3:
    pipe = build_pipeline(candidates[nombre])
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, proba)
    auc = roc_auc_score(y_te, proba)
    plt.plot(fpr, tpr, linewidth=2, label=f'{nombre} (AUC={auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Aleatorio')
plt.xlabel('Tasa de falsos positivos (1 - especificidad)')
plt.ylabel('Tasa de verdaderos positivos (sensibilidad)')
plt.title('ROC — top 3 modelos')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()
```

Nota: algunos modelos pueden ganar en la zona de alta sensibilidad pero perder en alta especificidad (y viceversa). Ese es el tipo de decisión que termina importando cuando eliges un umbral clínico.

#### 7.3 Tu tarea

El estudiante debe:

1. Correr el bloque anterior tal cual
2. **Modificar al menos 3 modelos del zoo** ajustando hiperparámetros (cambiar `C`, `max_depth`, `n_estimators`, `class_weight`, etc.)
3. Opcionalmente, crear nuevos modelos agregando entradas al dict
4. Elegir el ganador y justificar

Para bajar fricción aún más, el notebook incluye una **guía de qué hiperparámetros mueven la aguja en cada modelo:**

| Modelo | Hiperparámetros que valen la pena tunear |
|---|---|
| Logística | `C` (0.01 → 10), `penalty` ('l1' vs 'l2'), `class_weight` |
| Árbol | `max_depth` (3 → 15), `min_samples_leaf` (1 → 50) |
| Random Forest | `n_estimators` (100 → 500), `max_depth` (5 → 20), `min_samples_leaf` |
| SVM lineal | `C` (0.01 → 10), `class_weight` |

**Reglas de la competencia (markdown claro):**

> **Tu misión:** construir el mejor modelo posible para predecir readmisión a 30 días en el validation set. Los mejores AUC recibirán mejor nota en este componente (la escala exacta se publicará cuando veamos los resultados).
>
> **Lo que puedes hacer:**
> - Modificar hiperparámetros de los modelos del zoo
> - Agregar nuevos modelos al zoo (cualquier clasificador sklearn)
> - Probar `VotingClassifier` combinando varios modelos
> - Jugar con `class_weight='balanced'` en modelos desbalanceados
>
> **Lo que NO deberías hacer:**
> - Buscar el dataset original en internet para hacer lookup del outcome (deshonesto, y se detecta porque tu AUC sería ~1.0)
> - Usar modelos fuera del ecosistema sklearn estándar (no XGBoost, LightGBM, CatBoost, ni redes neuronales)
>
> **Nota sobre múltiples entregas:** solo tu archivo final subido cuenta. Documenta tu proceso en el ejercicio 7.1 — eso es lo que te hace aprender, más que el número final.
>
> **Entrega:** al final, ejecuta `export_predictions(tu_mejor_modelo, X_val, 'TuApellido')` y sube el archivo generado junto con el notebook.

#### 7.4 Función de exportación (pre-llenada)

```python
def export_predictions(model, X_validation, student_lastname):
    """
    Aplica el modelo al validation set y exporta probabilidades.
    Requisito: model ya debe estar entrenado (.fit llamado) sobre X, y completos.
    """
    # Asegura que X_validation tiene encounter_id
    if 'encounter_id' not in X_validation.columns:
        raise ValueError("validation_features.csv debe tener columna encounter_id")

    # Separa features de id
    ids = X_validation['encounter_id'].values
    X_val_features = X_validation.drop(columns=['encounter_id'])

    proba = model.predict_proba(X_val_features)[:, 1]

    out = pd.DataFrame({
        'encounter_id': ids,
        'prob_readmission': proba
    })
    filename = f'predictions_{student_lastname.lower()}.csv'
    out.to_csv(filename, index=False)

    print(f'Guardado: {filename}')
    print(f'Filas: {len(out)}')
    print(f'Probabilidades — min: {proba.min():.3f}, max: {proba.max():.3f}, '
          f'media: {proba.mean():.3f}')

    # Sanity checks
    assert len(out) == 2000, f"Error: esperaba 2000 filas, hay {len(out)}"
    assert not out['prob_readmission'].isna().any(), "Error: hay NaN en probs"
    assert (out['prob_readmission'] >= 0).all() and (out['prob_readmission'] <= 1).all(), \
        "Error: probs fuera de [0,1]"
    print("\n✓ Archivo válido para entrega.")
    return out
```

#### 7.5 Ejercicios de cierre de la Parte 7

**Ejercicio 7.1 (markdown — proceso documentado):**
> Documenta tu proceso. Mínimo 3 modelos distintos comparados. Incluye:
> - Qué probaste y qué cambiaste
> - Qué funcionó mejor y tu hipótesis de por qué
> - Qué descartaste y por qué
>
> Esto no se evalúa numéricamente pero es obligatorio para que la entrega sea válida.

**Ejercicio 7.2 (markdown — recomendación clínica, 200–400 palabras):**
> Eres científico de datos del Hospital Roosevelt. Escribe una recomendación para el comité de seguridad del paciente:
> - ¿Qué modelo propones usar para tamizaje de readmisión?
> - ¿Qué umbral de clasificación eliges y por qué? (considera costo de FN vs FP)
> - ¿Qué limitaciones debe conocer el comité?
> - ¿Qué harías antes de desplegarlo en producción?

### Parte 8: Reflexión (libre — ~15 min)

Markdown con respuestas:

8.1. ¿Tu modelo ganador es **significativamente** mejor que la logística baseline, o la diferencia está dentro del ruido de CV (compara con las desviaciones estándar)?

8.2. **No Free Lunch:** describe una situación clínica donde el algoritmo que elegiste sería **la peor opción**.

8.3. **Dato vs modelo:** ¿qué crees que movería más la aguja: pasar tu AUC de 0.63 a 0.65 con mejor tuning, o conseguir 20 variables clínicas adicionales de buena calidad?

8.4. **Validación externa:** el modelo fue entrenado con 130 hospitales de EE.UU. (1999–2008). ¿Qué anticipas si lo aplicas en Guatemala hoy?

8.5. **Conexión con laboratorios futuros:** ¿cómo monitorearías este modelo en producción? (pensamiento libre, no se requiere conocer Grafana aún)

---

## 6. Rúbrica de evaluación

**Decisión:** la escala exacta se define después de ver los resultados del primer corrido. Por ahora:

- Lab vale 1 punto (como todos los labs)
- Cumplir entregables válidamente otorga la nota base
- Los mejores AUC del validation set reciben bono/nota superior
- La escala se calibra post-hoc según distribución real de resultados

**Entregables mínimos (para que valga el lab):**

- [ ] Notebook corrido end-to-end sin errores
- [ ] Todos los ejercicios completados (incluidos los markdown)
- [ ] Ejercicio 7.1 (proceso documentado) con mínimo 3 modelos comparados
- [ ] Ejercicio 7.2 (recomendación clínica) con 200+ palabras
- [ ] Archivo `predictions_<apellido>.csv` válido (2000 filas, probabilidades en [0,1], sin NaN)

**Criterios de entrega inválida (no cuenta el lab):**
- Archivo mal nombrado
- Número de filas ≠ 2,000
- Falta columna `prob_readmission` o `encounter_id`
- NaN/Inf en probabilidades
- Probabilidades fuera de [0,1]
- Falta el ejercicio 7.1

---

## 7. Estilo y convenciones

Seguir exactamente el estilo de Lab 4:

- Título H1 con metadata del curso
- Introducción, objetivos de aprendizaje numerados (5–6 items), tabla de estructura
- Cada parte: `## Parte N: Nombre`
- Sub-secciones: `### N.1`, `### N.2`
- Ejercicios: `**Ejercicio N.M:**`
- Preguntas abiertas: `**Pregunta N.M:**` seguido de `*Tu respuesta aquí...*`
- Checkpoints: bloque con `git add -A && git commit -m "Checkpoint N: ..."`

**Tono:**
- Español académico pero cercano
- Explicaciones claras antes del código técnico
- Sin hype, sin emojis (salvo los que Lab 4 ya usa)
- Cuando una decisión técnica se simplifica, explicar por qué (ej: "usamos LinearSVC en vez de SVC-RBF porque en datos tabulares grandes el entrenamiento se vuelve impráctico y la mejora en AUC no lo justifica")

**Importaciones estándar (primera celda):**

```python
# ── Datos y visualización ──
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Machine Learning ──
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, ConfusionMatrixDisplay, RocCurveDisplay
)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

print("Librerías cargadas correctamente.")
```

**Paleta:** `steelblue` y `coral`, gris para anotaciones. Igual que Lab 4.

---

## 8. Archivos a producir

1. **`prepare_dataset.py`** — script del instructor, corre una vez antes del curso. Descarga UCI con `ucimlrepo`, procesa, hace split por paciente, genera `public_train.csv`, `validation_features.csv`, `validation_ground_truth.csv`.

2. **`Lab06_Machine_Learning_Supervisado.ipynb`** — notebook principal del lab.

3. **`grade_submissions.py`** — script del instructor. Lee todos los `predictions_*.csv`, hace merge con ground truth por `encounter_id`, calcula AUC por estudiante, flaggea entregas inválidas.

4. **`README.md`** — siguiendo el patrón de Lab 4 (contexto, objetivo, qué se construye, flujo de trabajo, resultado esperado).

5. **`requirements.txt`** — `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `ucimlrepo` (este último solo para `prepare_dataset.py`, no para el notebook del estudiante).

6. **`docker-compose.yml`** — copiar de Lab 4 adaptando volumen y nombre de servicio.

7. **Carpeta `data/`** con `public_train.csv` y `validation_features.csv` pre-generados para que los estudiantes no necesiten correr `prepare_dataset.py`.

---

## 9. Casos borde a manejar en implementación

- **`ConvergenceWarning` de logística:** usar `max_iter=1000` o más en todos los ejemplos.
- **Balance de clases:** usar `class_weight='balanced'` por default en el model zoo. Explicar que cambia el threshold óptimo pero rara vez mejora AUC — es un error conceptual común que el notebook debe aclarar.
- **OHE con categorías nuevas en validation:** `handle_unknown='ignore'` ya lo resuelve.
- **Reproducibilidad:** todos los `random_state=42`.
- **Performance:** todos los `n_jobs=-1` donde aplique (cross_val_score, RF).
- **Encoding en CSVs:** `to_csv(..., encoding='utf-8')` explícito porque hay usuarios en Windows.

**En `grade_submissions.py`:**
- Match por `encounter_id`, no por orden de filas
- Detectar archivos mal nombrados, NaN/Inf, probabilidades fuera de [0,1]
- Intentar `clip` a [0,1] y flaggear antes de marcar inválido (por si solo son errores de precisión flotante)
- Robustez a BOM y encoding (Windows suele exportar UTF-8 con BOM)

---

## 10. Checklist del agente implementador

Antes de declarar el lab "hecho":

- [ ] `prepare_dataset.py` corre sin errores con `ucimlrepo` y genera los 3 CSVs esperados
- [ ] `public_train.csv` tiene ~20,000 filas con columnas `encounter_id` y `target`
- [ ] `validation_features.csv` tiene 2,000 filas con `encounter_id`, sin `target`
- [ ] `validation_ground_truth.csv` tiene 2,000 filas con `encounter_id` y `target`
- [ ] **Verificar que ningún `patient_nbr` aparece en ambos splits** (ver output del script)
- [ ] El notebook corre end-to-end sin errores ni warnings visibles
- [ ] `build_pipeline()` y `evaluate()` funcionan con todos los modelos del zoo
- [ ] `plot_tree` del árbol visual es legible (nombres de features claros, máximo 3 niveles)
- [ ] Ejercicio 5.1 (verificar predict_proba manualmente) imprime números que efectivamente coinciden (`np.isclose` devuelve True)
- [ ] Tiempos tolerables: logística <20s, RF-200 <30s, SVM-lineal <60s
- [ ] Logística baseline alcanza AUC > 0.60 en CV (confirma que el pipeline funciona)
- [ ] `export_predictions` genera CSV válido con los sanity checks
- [ ] `grade_submissions.py` procesa correctamente archivos de ejemplo, incluidos casos inválidos
- [ ] Tono y estilo consistentes con Lab 4
- [ ] Referencias al final: UCI dataset, paper Strack 2014, Breiman 2001 (RF), Cortes & Vapnik 1995 (SVM)

---

**Fin de la especificación.**
