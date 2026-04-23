"""Genera Lab06_solutions.ipynb a partir del notebook del estudiante.

Reemplaza los marcadores `*Tu respuesta aquí...*` con respuestas-modelo que
sirven como referencia para el instructor al calificar. Las respuestas no
son únicas — los estudiantes pueden llegar a conclusiones distintas y
seguir siendo correctos.

Uso:
    python instructor/build_solutions_notebook.py
"""
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB_ROOT = HERE.parent
NOTEBOOKS_DIR = LAB_ROOT / "notebooks"
STUDENT_NB = NOTEBOOKS_DIR / "Lab06_Machine_Learning_Supervisado.ipynb"
SOLUTIONS_NB = HERE / "Lab06_solutions.ipynb"


# Mapeo de preguntas a respuestas-modelo. Las keys son sub-strings únicos
# que aparecen en el markdown ANTERIOR al *Tu respuesta aquí...*. El script
# encuentra el próximo cell "*Tu respuesta aquí...*" después de cada match
# y lo reemplaza.
ANSWERS = {
    # Ejercicio 1.1 — accuracy vs AUC en desbalance
    "¿Por qué **AUC** es más informativa que accuracy": """**Respuesta modelo (referencia):**

Con solo 11% de positivos, un clasificador trivial que siempre predice "no readmitido" alcanza **89% de accuracy** sin identificar a ningún paciente de riesgo — totalmente inútil clínicamente. La accuracy premia al modelo por acertar en la clase mayoritaria, que en este caso es la menos interesante.

AUC mide la capacidad de **rankear** correctamente (dado un par positivo–negativo, ¿el modelo asigna mayor probabilidad al positivo?). Es independiente del umbral y del balance de clases, por lo que distingue un modelo útil de uno trivial incluso cuando la prevalencia es baja.""",

    # Ejercicio 3.1 — piso AUC baseline
    "Anota el AUC de CV que obtuviste con la logística base": """**Respuesta modelo (referencia):**

AUC baseline esperado: **≈ 0.63 ± 0.01**. Cualquier modelo más complejo que no supere este número no justifica su costo computacional.""",

    # Ejercicio 4.1 — follow patient (solo confirmación)
    # No hay markdown de respuesta aquí — es ejercicio de ejecución

    # Pregunta 4.3 — bias-variance
    "conecta lo que observaste con el concepto de **bias-variance**": """**Respuesta modelo (referencia):**

- `max_depth=3` → **alto sesgo**: el árbol es demasiado simple para capturar la estructura del problema. Train y CV dan AUC similarmente bajo.
- `max_depth=8` → **sweet spot**: complejidad suficiente sin sobreajustar. AUC de CV óptimo.
- `max_depth=None` → **alta varianza**: el árbol memoriza el train (AUC ≈ 1.0 en train) pero no generaliza (AUC de CV cerca de 0.52, apenas mejor que azar).

El patrón del bias-variance trade-off: al aumentar la complejidad del modelo, el sesgo baja pero la varianza sube. El modelo óptimo está donde la suma de ambos errores es mínima.""",

    # Post-Ej 5.1 — predict_proba verification
    "¿Qué te enseña este experimento sobre cómo \"vota\" un Random Forest?": """**Respuesta modelo (referencia):**

`RandomForestClassifier.predict_proba` calcula el **promedio aritmético** de `tree.predict_proba` sobre todos los árboles. No es un voto por mayoría sobre la clase predicha, sino un promedio de las probabilidades que cada árbol asigna. Este mecanismo suaviza las predicciones: si 60 árboles dan 0.51 y 40 árboles dan 0.49, el forest predice 0.504 — una señal débil — aunque una "mayoría" técnicamente vota por la clase positiva.""",

    # Pregunta 5.3 — feature importance interpretation
    "¿Tienen sentido clínico para predecir readmisión a 30 días?": """**Respuesta modelo (referencia):**

Típicamente aparecen en el top: `number_inpatient`, `discharge_disposition_id`, `number_diagnoses`, `time_in_hospital`, `num_medications`.

Las tres primeras tienen alto sentido clínico:
- **number_inpatient** — hospitalizaciones previas son el predictor más fuerte de readmisión futura (patrón bien documentado en literatura)
- **discharge_disposition_id** — el destino al alta (casa, centro de rehabilitación, hospicio) predice fuertemente la transición y el riesgo de rehospitalización
- **number_diagnoses** — más comorbilidades = paciente más complejo = mayor riesgo

Los otros (`time_in_hospital`, `num_medications`) pueden actuar como proxies de severidad/complejidad del caso más que factores causales. El estudiante debe notar que "importante para el modelo" ≠ "causal".""",

    # Post-Ej 6.1 — intuition of C
    "¿Qué hace `C` intuitivamente?": """**Respuesta modelo (referencia):**

`C` regula el trade-off entre margen amplio y errores en train. `C` alto = el modelo prioriza clasificar correctamente todos los puntos de entrenamiento (riesgo de overfit, hiperplano menos robusto). `C` bajo = el modelo tolera más errores individuales a cambio de un margen más amplio (mejor generalización).""",

    # Ejercicio 7.1 — proceso documentado
    "Esto no se evalúa numéricamente pero es **obligatorio**": """**Respuesta modelo (ejemplo de documentación aceptable — 150+ palabras, 3 modelos):**

Probé tres líneas principales de modelos:

1. **Logística L2 con `C` variable (0.01, 0.1, 1, 10):** el mejor fue `C=1` con AUC 0.630 ± 0.009. Aumentar `C` no mejoró — la logística ya está en su techo en este problema.

2. **Random Forest con profundidades variables (5, 10, 15, None):** `max_depth=10, n_estimators=200` dio AUC 0.638 ± 0.012. Aumentar `n_estimators` a 500 no movió la aguja (0.639). Profundidades sin límite bajaron el AUC por overfit.

3. **Voting ensemble (logística + RF + SVM lineal):** AUC 0.635, ligeramente inferior al mejor RF. El ensemble no ayudó porque los modelos base son demasiado correlacionados — todos usan las mismas features y capturan las mismas señales.

Descarté `class_weight='balanced'` porque aunque cambia la calibración, no mejoró el AUC. Mi modelo final es Random Forest con 200 árboles y `max_depth=10`. Gané ~0.008 AUC sobre la logística baseline — significativo pero pequeño en términos prácticos.""",

    # Ejercicio 7.2 — recomendación clínica
    "¿Qué harías antes de desplegarlo** en producción?": """**Respuesta modelo (ejemplo de recomendación clínica aceptable — 200+ palabras):**

Para el comité de seguridad del paciente del Hospital Roosevelt:

**Modelo propuesto:** Random Forest (200 árboles, `max_depth=10`) entrenado sobre 20,000 encuentros del UCI Diabetes 130-US Hospitals. AUC 0.638 ± 0.012 en validación cruzada — mejor que el azar (0.5) pero lejos de ser perfecto.

**Umbral de clasificación:** propongo umbral bajo (0.15) para privilegiar sensibilidad sobre especificidad. El costo de un falso negativo (dar de alta a un paciente que será readmitido sin asignar recursos de seguimiento) es mayor que un falso positivo (seguimiento innecesario en un paciente de bajo riesgo). Con este umbral el sistema actuaría como **tamizaje**, no como decisión final — un trabajador social revisa los casos flaggeados.

**Limitaciones a comunicar:**
- El modelo fue entrenado con datos de EE.UU. entre 1999–2008. Casi todas las variables son administrativas (ICD-9, medicamentos genéricos). No incluye información social (vivienda, soporte familiar) que es altamente predictiva en nuestro contexto.
- AUC 0.64 implica que el modelo se equivoca con frecuencia. No debe usarse como criterio único de decisión clínica.
- El dataset tiene ~11% de positivos. Las métricas absolutas (sensibilidad/especificidad al umbral elegido) importan más que el AUC solo.

**Antes de desplegar en producción:**
1. **Validación externa** con 500+ encuentros del Roosevelt — verificar que el AUC se mantiene.
2. **Recalibración** para que las probabilidades reflejen riesgo real en nuestra población.
3. **Integración con trabajo social** — el modelo identifica candidatos; un humano decide la intervención.
4. **Monitoreo continuo** — tasa de readmisión observada vs predicha, estratificada por subgrupos (edad, género).
5. **Evaluación ética** con el comité de bioética sobre equidad del modelo entre subgrupos.""",

    # Pregunta 8.1 — ¿ganaste realmente?
    "o la diferencia está dentro del ruido de CV": """**Respuesta modelo (referencia):**

Respuesta típica: el RF ganador da ~0.638 ± 0.012 y la logística da ~0.630 ± 0.009. La diferencia de medias (~0.008) es menor que las desviaciones estándar, por lo que los intervalos se superponen — **la mejora no es estadísticamente significativa** en este problema. El estudiante debe notar que una "victoria" en AUC no siempre implica superioridad robusta.""",

    # Pregunta 8.2 — No Free Lunch
    "Describe una situación clínica donde el algoritmo que elegiste sería **la peor opción**": """**Respuesta modelo (referencia, varía según el modelo ganador):**

- Si el estudiante eligió **Random Forest**: peor opción cuando la interpretabilidad es requisito legal/clínico (diagnóstico con justificación al paciente, auditoría regulatoria). Un médico no puede explicar por qué 200 árboles votaron de cierta forma.
- Si eligió **logística**: peor opción cuando hay interacciones no-lineales fuertes entre features (p. ej. la decisión depende de la combinación edad × medicamento × función renal simultáneamente). La logística asume aditividad.
- Si eligió **SVM**: peor opción con datasets muy grandes (millones de filas) — escala cuadráticamente.""",

    # Pregunta 8.3 — dato vs modelo
    "conseguir 20 variables clínicas adicionales de buena calidad": """**Respuesta modelo (referencia):**

La opción **B** (mejores datos) mueve mucho más la aguja. Pasar de AUC 0.63 a 0.65 con tuning es ~2 puntos que probablemente están dentro del ruido. Agregar variables como "soporte social al alta", "medicamentos en casa", "estado funcional", "labs al día del alta" puede subir el AUC más allá de 0.75 porque son señales que no están en el dataset actual. La regla práctica: cuando el techo del AUC no mueve con más tuning, el problema no es el modelo — es la señal.""",

    # Pregunta 8.4 — validación externa
    "¿Qué problemas técnicos puedes anticipar?": """**Respuesta modelo (referencia):**

**Técnicos:**
- Distribuciones distintas: edad, prevalencia de diabetes, tipos de admisión en Guatemala difieren de EE.UU. 1999–2008.
- Categorías nuevas en variables categóricas (especialidades médicas, medicamentos) que OneHotEncoder maneja como `unknown`.
- Missing patterns distintos: en Guatemala quizás faltan labs que en EE.UU. son rutina, y viceversa.
- Codificación: el dataset original usa ICD-9, pero Guatemala hoy codifica en ICD-10.

**Clínicos:**
- Case-mix distinto: pobreza, acceso a medicamentos, seguro vs IGSS vs privado, rural vs urbano cambian el perfil de los pacientes.
- Protocolos de alta diferentes: qué cuenta como "readmisión" puede depender del hospital.
- Estructura del sistema: EE.UU. tiene Medicare, Guatemala tiene mezcla pública/privada. La definición operativa de "admisión" puede no ser comparable.

**Qué haría:** validar sobre una muestra del Roosevelt antes de desplegar; considerar re-entrenar con datos locales si hay suficiente volumen.""",

    # Pregunta 8.5 — monitoreo en producción
    "indicarían que el modelo está **degradándose**": """**Respuesta modelo (referencia):**

**Métricas a monitorear:**
- **AUC mensual** sobre un holdout prospectivo (encuentros nuevos con outcome observado a 30 días).
- **Calibración**: probabilidades predichas vs tasas observadas de readmisión por bucket (0–10%, 10–20%, ...).
- **Distribución de features en producción** vs train: si cambia marcadamente, hay *data drift*.
- **Sensibilidad y especificidad al umbral fijo** elegido.

**Señales de degradación:**
- Caída sostenida de AUC >0.03 durante 3 meses consecutivos
- Calibración rota: los buckets de "alta probabilidad" ya no corresponden a alta tasa de readmisión observada
- Shift en distribución de features (ej. nuevas comorbilidades por un cambio epidemiológico)
- Nuevos patrones de missing (ej. el hospital cambió de EHR y ciertas columnas ya no se registran igual)

**Respuesta:** re-entrenar con los datos más recientes, o al menos recalibrar. El patrón es el mismo que veremos en Lab 10 / Grafana: si la distribución de lo que entra al modelo cambia, el modelo tiene que adaptarse.""",
}


def _src(cell: dict) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _set_src(cell: dict, text: str) -> None:
    lines = text.split("\n")
    cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]


RESPONSE_MARKER_PREFIX = "*Tu respuesta aquí"


def _has_marker(cell: dict) -> bool:
    """True si el cell es markdown con un marcador de respuesta."""
    if cell["cell_type"] != "markdown":
        return False
    return RESPONSE_MARKER_PREFIX in _src(cell)


def main() -> None:
    if not STUDENT_NB.exists():
        raise SystemExit(f"No encuentro el notebook del estudiante: {STUDENT_NB}")

    with open(STUDENT_NB, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = copy.deepcopy(nb["cells"])

    replaced = 0
    for anchor, answer in ANSWERS.items():
        # Busca el cell que contiene el anchor
        anchor_idx = -1
        for i, cell in enumerate(cells):
            if cell["cell_type"] != "markdown":
                continue
            if anchor in _src(cell):
                anchor_idx = i
                break
        if anchor_idx == -1:
            print(f"  [WARN] no encontré anchor: '{anchor[:50]}...'")
            continue

        anchor_src = _src(cells[anchor_idx])

        # Caso 1: el marker está en el mismo cell que el anchor → append in-place
        if RESPONSE_MARKER_PREFIX in anchor_src:
            _set_src(cells[anchor_idx], f"{anchor_src.rstrip()}\n\n{answer}")
            replaced += 1
            continue

        # Caso 2: el marker está en un cell INMEDIATAMENTE siguiente
        # (solo miramos el próximo cell para evitar sobrescribir cells distantes)
        if anchor_idx + 1 < len(cells) and _has_marker(cells[anchor_idx + 1]):
            resp_idx = anchor_idx + 1
            resp_src = _src(cells[resp_idx])
            _set_src(cells[resp_idx], f"{resp_src.rstrip()}\n\n{answer}")
            replaced += 1
            continue

        # Caso 3: no hay marker de respuesta — append al cell del anchor
        _set_src(cells[anchor_idx], f"{anchor_src.rstrip()}\n\n{answer}")
        replaced += 1

    nb["cells"] = cells

    # Ajusta el título para dejar claro que es la versión de soluciones
    first_cell = cells[0]
    first_src = "".join(first_cell["source"])
    first_src = first_src.replace(
        "# Lab 6 — Machine Learning Supervisado: Comparación de Algoritmos",
        "# Lab 6 — Machine Learning Supervisado (VERSIÓN INSTRUCTOR con respuestas modelo)"
    )
    lines = first_src.split("\n")
    first_cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]

    SOLUTIONS_NB.parent.mkdir(parents=True, exist_ok=True)
    with open(SOLUTIONS_NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"\nSolutions notebook generado: {SOLUTIONS_NB.relative_to(LAB_ROOT)}")
    print(f"Respuestas reemplazadas: {replaced}/{len(ANSWERS)}")


if __name__ == "__main__":
    main()
