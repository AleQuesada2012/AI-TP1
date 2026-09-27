# Figuras

Guardar aquí las figuras finales con nombres descriptivos y estables. No usar
nombres como `plot1.png` o `final_final.png`.

Las figuras PNG de esta carpeta las genera automáticamente el notebook
(`utils.configurar_exportacion_figuras` en la celda de configuración): cada
figura mostrada se guarda con un nombre derivado de su título, y las de la
sección de evaluación llevan el prefijo `evaluacion-`. Al ejecutar el notebook
se eliminan los PNG anteriores, así que no se deben editar a mano. El informe
LaTeX las toma desde aquí mediante `\graphicspath`.
