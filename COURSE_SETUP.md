# 📘 Trabajo con GitHub (Setup Único)

Este curso usa GitHub como se usa en equipos reales de datos y software.
No buscamos que memorices comandos, sino que **entiendas dónde estás parado** y **qué flujo seguir**.

---

## 🧱 El modelo mental

En este curso existen **tres niveles distintos**.

---

## 1️⃣ Repositorio del curso (fuente de verdad)

- Es el repositorio oficial de la clase
- Lo mantiene **el profesor**
- Ahí:
  - Aparecen nuevos laboratorios
  - Se corrigen instrucciones
  - Se define la estructura del curso

👉 **Nunca trabajas directamente aquí**

A este repositorio lo llamamos:

```
upstream
```

Piensa en él como:

> “La versión oficial del curso”

---

## 2️⃣ Fork del grupo (repositorio de trabajo)

- Cada grupo tiene **un fork** del repositorio del curso
- Un fork es una **copia completa del repositorio**, en la cuenta del grupo
- Ahí el grupo:
  - Crea branches
  - Hace commits
  - Abre Pull Requests (PRs)

👉 A este repositorio lo llamamos:

```
origin
```

Piensa en él como:

> “Nuestra copia del curso para trabajar”

> ⚠️ **Fork ≠ Branch**
>
> - Fork = repositorio completo
> - Branch = línea de trabajo dentro de un repositorio

---

## 3️⃣ Tu carpeta local (tu computadora)

- Es la carpeta que tienes en tu PC
- Ahí:
  - Editas archivos
  - Corres Docker
  - Abres Jupyter

- Esa carpeta está conectada a:
  - `origin` → fork del grupo
  - `upstream` → repo del curso

> 🔑 **Regla clave:**
> 👉 **No vuelvas a clonar el repositorio**
> Siempre trabajaremos sobre **la misma carpeta local**.

---

## 🧭 Cómo saber dónde estás parado (comandos esenciales)

### Ver en qué branch estás

```bash
git branch
```

El branch actual aparece con `*`.

---

### Ver a qué repositorio estás conectado

```bash
git remote -v
```

Deberías ver algo como:

- `origin` → fork del grupo
- `upstream` → repo del curso

Si no ves `upstream`, lo agregas **una sola vez**:

```bash
git remote add upstream URL_DEL_REPO_DEL_CURSO
```

---

## 🌿 Regla central del curso

> **Un fork por grupo · un branch por laboratorio · un PR por entrega**

---

## 🧪 Flujo estándar para CADA laboratorio

---

### 1️⃣ Actualizar instrucciones del curso

Antes de empezar un lab nuevo:

```bash
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```

👉 Esto trae **nuevos laboratorios** sin borrar tu trabajo previo.

---

### 2️⃣ Crear el branch del laboratorio

Desde `main`:

```bash
git checkout -b labX-nombre/grupo-N
```

Ejemplo:

```bash
git checkout -b lab1.1-veracity/grupo-4
```

> [!IMPORTANT]
> **Nunca trabajes en `main`.**

---

### 3️⃣ Trabajar solo en ese branch

- Edita archivos
- Corre Docker
- Ejecuta notebooks

Revisa siempre:

```bash
git status
```

---

### 4️⃣ Commits (durante el lab)

```bash
git add archivo.sql notebook.ipynb
git commit -m "Checkpoint: descripción clara"
```

> 🧠 Commit = guardar progreso
> ❌ Commit ≠ entrega final

---

### 5️⃣ Push al fork del grupo

```bash
git push -u origin labX-nombre/grupo-N
```

---

### 6️⃣ Pull Request (ENTREGA)

- Base: `main` (del repo del curso)
- Compare: `labX-nombre/grupo-N`
- **No hacer merge**

👉 El PR es la **entrega oficial**.

---

## 👥 Cómo trabajar en grupo (sin caos)

- **Un fork por grupo**
- **Un branch por laboratorio**
- **Una persona hace los commits y PR**
- Las otras personas:
  - Revisan
  - Ayudan
  - Aprenden el flujo

---

## 🧹 Carpetas “sucias” (problema común)

Si ya tienes el repo clonado y con cambios viejos:

✅ **NO vuelvas a clonar**

La solución correcta es:

- Usar **la misma carpeta**
- Asegurarte de:
  - estar en el branch correcto
  - tener `origin` y `upstream` bien configurados

Esto evita:

- múltiples carpetas
- repos dentro de repos
- archivos duplicados

---

## 🚨 Errores comunes y cómo evitarlos

### ❌ “No veo el branch de mis compañeros”

→ Están en forks distintos

✔️ Solución:
Un fork por grupo, no por persona.

---

### ❌ “Hice PR pero nadie lo ve”

→ PR hecho a tu propio fork

✔️ Solución:
PR **desde el fork del grupo → repo del curso**

---

### ❌ “Perdí lo del lab pasado”

→ No usaste branches

✔️ Solución:
Cada lab vive en su propio branch.

---

## 🧠 Qué estás aprendiendo realmente

Aunque no sea un curso de Git, aquí estás aprendiendo:

- cómo se trabaja en equipo
- cómo se versionan datos y análisis
- cómo se reproducen resultados
- cómo se colabora sin pisarse

Esto **sí importa** en ciencia de datos biomédicos real.

---

## ✅ Checklist rápido antes de trabajar

Antes de empezar cualquier lab:

- [ ] Estoy en la carpeta correcta
- [ ] `git branch` muestra el branch del lab
- [ ] `git remote -v` muestra origin y upstream
- [ ] No estoy en `main`
