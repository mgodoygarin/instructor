# Lab 3.1 — Datos Sintéticos con Synthea

En los labs anteriores:

- Construimos un EHR relacional (patients, admissions, diagnoses, labevents)
- Normalizamos su semántica con ICD-10 y LOINC
- Convertimos datos clínicos a FHIR y ensamblamos Bundles de intercambio

Ahora la pregunta natural:

> Tenemos el estándar FHIR. ¿Cómo generamos datos clínicos realistas a escala para investigación, pruebas y entrenamiento de modelos?

---

## 🎯 Objetivo del laboratorio

Usar **Synthea**, un simulador de poblaciones de pacientes sintéticos open-source, para:

- Entender por qué los datos sintéticos importan en salud digital
- Explorar la estructura de salida de Synthea (CSV y FHIR R4)
- Analizar una población sintética: demografía y epidemiología
- Comparar un Bundle FHIR generado por Synthea con el que construiste en Lab 3

---

## 🧱 Qué vamos a analizar

```
┌────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  Módulos de        │     │  Motor Synthea        │     │  Salida:            │
│  enfermedad        │────▶│  (simulación          │────▶│  CSV + FHIR R4      │
│  (JSON / máquinas  │     │   epidemiológica      │     │  por paciente       │
│   de estado)       │     │   probabilística)     │     │                     │
└────────────────────┘     └──────────────────────┘     └─────────────────────┘
```

Archivos CSV incluidos en `data/`:

| Archivo            | Contenido                        | Codificación   |
|--------------------|----------------------------------|----------------|
| `patients.csv`     | Demografía del paciente          | —              |
| `conditions.csv`   | Diagnósticos                     | **SNOMED-CT**  |
| `encounters.csv`   | Visitas y encuentros             | SNOMED-CT      |
| `medications.csv`  | Medicamentos prescritos          | **RxNorm**     |
| `observations.csv` | Signos vitales y laboratorio     | **LOINC**      |

---

## 📊 Qué aprenderás

- Por qué los datos sintéticos son esenciales para investigación biomédica
- Cómo Synthea modela trayectorias clínicas con máquinas de estado
- Cómo SNOMED-CT, LOINC y RxNorm se usan en datos generados (conexión directa con Lab 2)
- Análisis epidemiológico básico: prevalencias y comorbilidades
- Cómo un Bundle FHIR generado automáticamente compara con uno construido a mano (Lab 3)

---

## 🧭 Flujo de trabajo

Sigue el flujo definido en `COURSE_SETUP.md`.

1. Actualiza `main`
2. Crea tu branch:

```bash
git checkout -b lab31-synthetic-data/grupo-N
```

3. Trabaja únicamente en ese branch
4. Haz commits durante el proceso
5. Entrega con Pull Request

---

## 📓 Importante

Este laboratorio se guía completamente desde el notebook:

```
notebooks/Lab031_Datos_Sinteticos_Synthea.ipynb
```

Los datos sintéticos están incluidos en `data/` y `fhir/`.
**No necesitas instalar Java ni correr Synthea.**

---

## ✅ Antes de empezar

Verifica:

```bash
git branch
git remote -v
docker compose up -d
```

- No estás en `main`
- Tienes `origin` y `upstream`
- Docker está corriendo (jupyter)

---

## 🔁 (Opcional) Regenerar los datos

Los datos en `data/` fueron generados con el siguiente comando:

```bash
# Desde el directorio de Synthea
./run_synthea -s 42 -p 100 \
  --exporter.csv.export=true \
  --exporter.fhir.export=true \
  Massachusetts
```

Si quieres generar tu propia población (cambia la semilla `-s` o el tamaño `-p`):

```bash
# Con Docker (si no tienes Java instalado)
docker run --rm \
  -v "$(pwd):/synthea" \
  -w /synthea \
  gradle:8-jdk17 \
  bash -c "./gradlew run -Params=\"['-s','99','-p','50','--exporter.csv.export=true','--exporter.fhir.export=true','Massachusetts']\""
```

---

## 🧠 Resultado esperado

Al final deberías poder:

- Explicar qué es Synthea y para qué se usa en biomedicina
- Interpretar la estructura de CSV de Synthea y conectarla con Labs 1 y 2
- Calcular prevalencias básicas en una población sintética
- Comparar un Bundle FHIR de Synthea con el que construiste en Lab 3
- Reflexionar sobre las limitaciones y usos éticos de datos sintéticos en salud
