# Guía de colaboración

## Flujo de Git

- `main` debe representar el estado integrado y revisado.
- Crear ramas cortas con prefijos como `feat/`, `docs/`, `test/` o `chore/`.
- Evitar ramas permanentes por estudiante.
- Mantener cada pull request enfocado en una sola tarea.
- Solicitar al menos una revisión antes de integrar cambios.

## Edición del notebook

El notebook es la única fuente de verdad para la implementación y se trata
como un archivo difícil de fusionar:

- Acordar una sola persona editora o integradora por sección en cada momento.
- Avisar al equipo antes de reorganizar, añadir o eliminar muchas celdas.
- No mantener implementaciones duplicadas en archivos Python externos.
- Reiniciar el kernel y ejecutar todas las celdas antes de integrar resultados.
- Evitar salidas voluminosas o transitorias durante el desarrollo.
- Conservar las tablas y figuras finales únicamente en un commit deliberado de
  preparación de entrega.

## Reproducibilidad

- Mantener imports, semilla, puntos iniciales y parámetros compartidos en una
  sola celda de configuración al inicio del notebook.
- Reutilizar exactamente los mismos puntos iniciales cuando el enunciado lo
  requiera.
- Registrar cambios de dependencias en `requirements.txt`.
- Nunca incluir credenciales, tokens, archivos `.env` ni claves de servicios.

## Revisión antes de integrar

- El notebook abre sin errores y conserva un orden de ejecución comprensible.
- Las pruebas unitarias relevantes se ejecutan dentro del notebook.
- Las funciones nuevas contienen documentación compatible con Doxygen.
- Los cambios relacionados se reflejan en el informe LaTeX cuando corresponde.
- No se agregaron cachés, entornos virtuales ni resultados sin revisar.
