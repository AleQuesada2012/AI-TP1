# TP1 - Optimización

Repositorio del Trabajo Práctico 1 del curso de Principios de Inteligencia
Artificial. Los experimentos se mantienen en un único notebook de Jupyter,
con funciones compartidas en `notebooks/utils.py`.

Este repositorio contiene la implementación completa en PyTorch del
descenso del gradiente, RMSProp y el enjambre de partículas, su calibración
con Optuna y la evaluación sobre las tres funciones del enunciado. El
notebook se apoya en `notebooks/utils.py` y, al ejecutarse, exporta sus
figuras a `results/figures/`, que el informe LaTeX de `docs/report/` usa
para responder el enunciado.

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

Se necesita Python 3.11 o superior; el notebook se validó con Python 3.14. Los
comandos usan el Python instalado en el sistema: `py` en Windows y `python3` en
macOS y Linux. Abrir una terminal en la raíz del repositorio; si al inicio de
la línea aparece `(.venv)`, ejecutar primero `deactivate`.

### macOS y Linux

```bash
python3 -m pip install -r requirements.txt
python3 -c "import torch, optuna, pandas, matplotlib; print('ok')"
```

Si `python3` es anterior a 3.11 (por ejemplo, el Python que incluye macOS),
instalar una versión reciente desde python.org y usar ese ejecutable (por
ejemplo, `python3.14`) en lugar de `python3`. Si `pip` responde
`externally-managed-environment` (Python del sistema en Debian, Ubuntu o
Homebrew), crear un entorno virtual con `python3 -m venv .venv`, instalar con
`.venv/bin/python -m pip install -r requirements.txt` y usar
`.venv/bin/python` en lugar de `python3` en los demás comandos; en Debian y
Ubuntu, `venv` requiere el paquete `python3-venv`. En macOS, PyTorch solo
publica instaladores para Apple Silicon.

### Windows PowerShell

```powershell
py -m pip install -r requirements.txt
py -c "import torch, optuna, pandas, matplotlib; print('ok')"
```

`requirements.txt` fija pandas 3.0.3 porque, con el Control inteligente de
aplicaciones de Windows 11, pandas 3.0.6 queda bloqueado al importarse ("An
Application Control policy has blocked this file").

En ambos bloques, el último comando debe imprimir `ok`.

Iniciar JupyterLab desde la raíz del repositorio:

```bash
# macOS y Linux
python3 -m jupyter lab
```

```powershell
# Windows PowerShell
py -m jupyter lab
```

Se usa `-m` porque el comando `jupyter` no siempre está en el PATH. Si el
navegador no se abre solo, copiar el enlace
`http://localhost:8888/lab?token=...` que aparece en la terminal.

Para ejecutar el notebook completo:

1. Abrir `notebooks/tp1_optimizacion.ipynb` y verificar que el kernel sea
   **Python 3 (ipykernel)**.
2. Elegir **Kernel → Restart Kernel and Run All Cells…**. La ejecución tarda
   alrededor de un minuto.
3. Revisar los resultados al final del notebook: calibración con Optuna,
   evaluación con los mejores hiperparámetros y mejores corridas.

Para detener JupyterLab, presionar `Ctrl+C` en la terminal. En las secciones
siguientes, si `jupyter` no está en el PATH, usar `py -m jupyter` (Windows) o
`python3 -m jupyter` (macOS y Linux).

## Documentación interna

Doxygen no consume el notebook directamente. Su fuente documentable se genera
de forma temporal y nunca se edita manualmente; `notebooks/utils.py` también
se incluye directamente:

```bash
jupyter nbconvert --to python notebooks/tp1_optimizacion.ipynb --output tp1_optimizacion --output-dir build/doxygen-source
doxygen Doxyfile
```

La salida HTML queda en `build/doxygen/html/`. Tanto la exportación temporal
como la documentación generada se excluyen de Git.

## Informe externo

Con una distribución de LaTeX y `latexmk` instalados, ejecutar desde la raíz:

```bash
./build_pdf.sh
```

El script descarta los archivos intermedios y deja únicamente el PDF generado
en `submissions/tp1_optimizacion.pdf`.

## Equipo

- Alejandro Quesada
- Manfred Azofeifa
- Jose Adrián Herrera
