# Laboratorio 01 — Captura en el EHR y Modelos de Datos

**PostgreSQL · Mini-MIMIC** 🏥

En este laboratorio trabajarás con un **modelo simplificado de un Electronic Health Record (EHR)** inspirado en la estructura real de **MIMIC-III**.
El foco **no es análisis**, sino **cómo se capturan y modelan los datos clínicos en la realidad** usando PostgreSQL.

> **Idea central**
> Los datos clínicos no viven en una sola tabla.
> Se capturan como **eventos** que ocurren dentro de **encuentros clínicos (admissions)**.

---

## 🎯 Objetivos de aprendizaje

Al finalizar este laboratorio serás capaz de:

- Diferenciar **paciente** vs **encuentro clínico**
- Entender por qué `admissions` es el **eje del modelo EHR**
- Modelar datos clínicos reales con tablas y relaciones en PostgreSQL
- Ejecutar consultas SQL básicas con sentido clínico
- Trabajar con **checkpoints y commits** como en un entorno profesional

---

## 📐 Esquema mini-MIMIC (subconjunto pedagógico)

Trabajaremos con 5 tablas:

| Tabla        | Qué representa                 |
| ------------ | ------------------------------ |
| `patients`   | Identidad longitudinal         |
| `admissions` | Encuentros hospitalarios (eje) |
| `diagnoses`  | Diagnósticos por admisión      |
| `d_labitems` | Diccionario de laboratorios    |
| `labevents`  | Resultados de laboratorio      |

> 🔑 **Regla de oro**
> Los datos clínicos **cuelgan de la admisión**, no directamente del paciente.

---

## ⚠️ Checkpoints del laboratorio (léelo antes de empezar)

Este laboratorio se trabaja por **checkpoints**.
No sigas avanzando si no cumples el checkpoint actual.

> [!IMPORTANT]
> Cuando el docente indique una parada, **detente** y verifica tu estado antes de continuar.

### Checkpoint 1 — Infraestructura lista

✔ Docker Desktop corriendo
✔ `docker ps` muestra `db` y `jupyter`
✔ Jupyter abre en el navegador

_(No hay commit en este checkpoint)_

---

### Checkpoint 2 — Modelo EHR creado (commit obligatorio)

✔ Tablas creadas sin errores en PostgreSQL

👉 Commit obligatorio:

```bash
Checkpoint 1: EHR schema created
```

---

### Checkpoint 3 — Datos clínicos capturados (commit obligatorio)

✔ Datos insertados correctamente
✔ No hay errores de `INSERT`

👉 Commit obligatorio:

```bash
Checkpoint 2: sample clinical data inserted
```

---

### Checkpoint 4 — Consulta clínica funcionando

✔ Consulta SQL devuelve una tabla en Jupyter

👉 Commit final del laboratorio (entrega)

---

## 0) Reglas del laboratorio

> [!IMPORTANT]
>
> - Usa el **mismo entorno del Lab 00**
> - **Nunca trabajes en `main`**
> - No instales herramientas nuevas
> - No uses CSV ni Excel
> - Si te atoras, **para y pregunta**

---

## 1) Preparación

### 1.1 Entrar al laboratorio

Desde la raíz del repositorio:

```powershell
cd labs/lab-1-ehr-capture
```

### 1.2 Levantar servicios

```powershell
docker compose up -d
docker ps
```

> [!TIP]
> Si no ves `db` y `jupyter`, **no sigas**. Estás antes del Checkpoint 1.

---

## 🔵 CHECKPOINT 1 — Infraestructura lista

> [!NOTE]
> Si Jupyter abre en `http://localhost:8888`, vas bien.

---

## 2) Crear el modelo EHR en PostgreSQL

### 2.1 Crear el archivo SQL

Crea el archivo:

```
sql/002_ehr_schema.sql
```

Contenido:

```sql
-- Patients: identidad longitudinal
CREATE TABLE IF NOT EXISTS patients (
  subject_id SERIAL PRIMARY KEY,
  sex CHAR(1) CHECK (sex IN ('M','F','O')),
  date_of_birth DATE,
  date_of_death DATE
);

-- Admissions: encuentros clínicos (eje del modelo)
CREATE TABLE IF NOT EXISTS admissions (
  hadm_id SERIAL PRIMARY KEY,
  subject_id INT REFERENCES patients(subject_id),
  admittime TIMESTAMP,
  dischtime TIMESTAMP,
  admission_type TEXT,
  hospital_expire_flag BOOLEAN
);

-- Diagnoses: diagnósticos por admisión
CREATE TABLE IF NOT EXISTS diagnoses (
  diagnosis_id SERIAL PRIMARY KEY,
  hadm_id INT REFERENCES admissions(hadm_id),
  diagnosis_text TEXT
);

-- Lab dictionary
CREATE TABLE IF NOT EXISTS d_labitems (
  labitem_id SERIAL PRIMARY KEY,
  label TEXT,
  unit TEXT
);

-- Lab events
CREATE TABLE IF NOT EXISTS labevents (
  labevent_id SERIAL PRIMARY KEY,
  hadm_id INT REFERENCES admissions(hadm_id),
  labitem_id INT REFERENCES d_labitems(labitem_id),
  charttime TIMESTAMP,
  value_num NUMERIC
);
```

### 2.2 Ejecutar el SQL

```powershell
Get-Content .\sql\002_ehr_schema.sql | docker compose exec -T db psql -U uvg_user -d health_data
```

---

## 🔵 CHECKPOINT 2 — Modelo EHR creado

> [!IMPORTANT]
> No sigas sin hacer este commit.

```powershell
git add sql/002_ehr_schema.sql
git commit -m "Checkpoint 1: EHR schema created"
```

---

## 3) Simular captura de datos clínicos

Agrega **al final del mismo archivo SQL**:

```sql
-- Patients
INSERT INTO patients (sex, date_of_birth) VALUES
('F', '1980-03-12'),
('M', '1975-07-01');

-- Admissions
INSERT INTO admissions (subject_id, admittime, dischtime, admission_type, hospital_expire_flag) VALUES
(1, '2101-01-10 08:00', '2101-01-15 14:00', 'Emergency', false),
(2, '2101-03-20 22:00', '2101-03-28 10:00', 'Emergency', true);

-- Diagnoses
INSERT INTO diagnoses (hadm_id, diagnosis_text) VALUES
(1, 'Hypertension'),
(2, 'Sepsis');

-- Lab dictionary
INSERT INTO d_labitems (label, unit) VALUES
('Creatinine', 'mg/dL'),
('Hemoglobin', 'g/dL');

-- Lab events
INSERT INTO labevents (hadm_id, labitem_id, charttime, value_num) VALUES
(1, 1, '2101-01-11 06:00', 1.2),
(1, 1, '2101-01-13 06:00', 1.8),
(2, 1, '2101-03-21 07:00', 2.5);
```

Vuelve a ejecutar el archivo SQL.

---

## 🔵 CHECKPOINT 3 — Datos clínicos capturados

```powershell
git add sql/002_ehr_schema.sql
git commit -m "Checkpoint 2: sample clinical data inserted"
```

> [!NOTE]
> Esto **simula captura clínica real**, no carga de datasets.

---

## 4) Consultas desde Jupyter

Abre `connection_test.ipynb`.

### 4.1 Ver pacientes y admisiones

```python
pd.read_sql("""
SELECT p.subject_id, a.hadm_id, a.admission_type, a.admittime, a.dischtime
FROM patients p
JOIN admissions a ON p.subject_id = a.subject_id
ORDER BY p.subject_id;
""", engine)
```

---

## 🔵 CHECKPOINT 4 — Consulta funcionando

> [!IMPORTANT]
> Si ves una tabla, **cerraste el ciclo DB → análisis**.

---

## 5) Consultas guiadas (escritas por ti)

### 5.1 ¿Cuántas admisiones tiene cada paciente?

Pistas:

- `COUNT(*)`
- `GROUP BY subject_id`

### 5.2 ¿Duración de estancia por admisión?

Pistas:

- `dischtime - admittime`
- alias como `length_of_stay`

### 5.3 Máximo valor de creatinina por admisión

Pistas:

- JOIN `labevents` + `d_labitems`
- filtra `Creatinine`
- usa `MAX(value_num)`

---

## 6) Reflexión (responde en el PR)

> [!CAUTION]
> Si no puedes responder esto, **no entendiste el modelo**, aunque el código funcione.

1. ¿Por qué los laboratorios no están en `patients`?
2. ¿Qué representa `hadm_id` clínicamente?
3. ¿Qué problema tendría una sola tabla?

---

## 7) Entrega (Pull Request)

```powershell
git checkout main
git pull
git checkout -b lab01-ehr-capture/grupo-XX
git add sql/002_ehr_schema.sql *.ipynb
git commit -m "Lab01: EHR capture model and clinical queries"
git push -u origin lab01-ehr-capture/grupo-XX
```

- Base: `main`
- Compare: `lab01-ehr-capture/grupo-XX`
- **No hacer merge**

---

## ✅ Checklist final

- [ ] Checkpoint 1 completo
- [ ] Checkpoint 2 + commit
- [ ] Checkpoint 3 + commit
- [ ] Consulta clínica funciona
- [ ] PR abierto

---
