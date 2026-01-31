-- Patients: identidad longitudinal
CREATE TABLE IF NOT EXISTS patients (
  subject_id SERIAL PRIMARY KEY,
  external_id TEXT UNIQUE,
  full_name TEXT,
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