-- Lab 1.1: Veracity / Missing Values
-- Insertamos datos "reales": campos faltantes y errores de dedo.

-- A) 3 pacientes "sucios" (missing + typo)
INSERT INTO patients (external_id, full_name, sex, date_of_birth) VALUES
('MRN-9001', 'Paciente Sin Fecha', 'F', NULL),            -- Missing DOB
('MRN-9002', 'Paciente Sin Nombre', 'M', '1990-01-01'),   -- Nombre vacío (lo simulamos después)
('MRN-9003', 'Paciente Ciudad Typo', 'M', '1985-09-10');  -- Ciudad con typo (simulada en admissions o diagnóstico)

-- B) Si tu tabla patients no tiene "full_name" nullable o quieres simular missing:
-- actualizamos a NULL un nombre (para simular realidad)
UPDATE patients
SET full_name = NULL
WHERE external_id = 'MRN-9002';

-- C) Simulamos un error de dedo en texto clínico (diagnosis)
-- Creamos una admisión para MRN-9003 y le ponemos un texto con typo
-- (Esto asume que subject_id de MRN-9003 existe. Si falla, revisa el Troubleshooting)
INSERT INTO admissions (subject_id, admittime, dischtime, admission_type, hospital_expire_flag)
SELECT subject_id, '2101-10-01 08:00', '2101-10-02 12:00', 'Emergency', false
FROM patients
WHERE external_id = 'MRN-9003';

INSERT INTO diagnoses (hadm_id, diagnosis_text)
SELECT a.hadm_id, 'Guateeeemala referral note'
FROM admissions a
JOIN patients p ON p.subject_id = a.subject_id
WHERE p.external_id = 'MRN-9003'
ORDER BY a.hadm_id DESC
LIMIT 1;