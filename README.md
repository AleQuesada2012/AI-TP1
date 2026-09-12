# TP1 - Optimización

Repositorio del Trabajo Práctico 1 del curso de Principios de Inteligencia
Artificial. La implementación se mantiene completamente dentro de un único
notebook de Jupyter.

Este repositorio contiene únicamente la estructura inicial del proyecto. No
incluye implementaciones, resultados ni respuestas al enunciado.

## Entregables

- Notebook autocontenido: `notebooks/tp1_optimizacion.ipynb`.
- Informe externo escrito en LaTeX: `docs/report/`.
- PDF final del informe, generado desde LaTeX.
- Documentación interna con formato Doxygen, generada desde una exportación
  temporal del notebook.

## Estructura

- `docs/assignment/`: enunciado original.
- `docs/report/`: fuentes LaTeX y bibliografía del informe.
- `docs/doxygen/`: portada y configuración auxiliar de Doxygen.
- `notebooks/`: notebook canónico y autocontenido.
- `results/`: figuras y tablas seleccionadas para el informe.
- `submission/`: copias finales preparadas exclusivamente para la entrega.

## Preparación del entorno

Antes de implementar, el equipo debe acordar una misma versión menor de
Python y registrar versiones de dependencias que hayan sido validadas en los
sistemas operativos utilizados.

### macOS y Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Iniciar JupyterLab desde la raíz del repositorio:

```bash
jupyter lab
```

## Documentación interna

Doxygen no consume el notebook directamente. La fuente documentable se genera
de forma temporal y nunca se edita manualmente:

```bash
jupyter nbconvert --to python notebooks/tp1_optimizacion.ipynb --output tp1_optimizacion --output-dir build/doxygen-source
doxygen Doxyfile
```

La salida HTML queda en `build/doxygen/html/`. Tanto la exportación temporal
como la documentación generada se excluyen de Git.

## Informe externo

Con una distribución de LaTeX y `latexmk` instalados, ejecutar desde la raíz:

```bash
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error -outdir=../../build/report docs/report/main.tex
```

Los productos intermedios permanecen en `build/`. El PDF revisado se copia a
`submission/` únicamente al preparar la entrega.

## Equipo

- Estudiante 1
- Estudiante 2
- Estudiante 3
