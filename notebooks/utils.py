import hashlib
import math
import numbers
import random
import re
import unicodedata
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from IPython.display import display
from matplotlib import patheffects
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullLocator

_figures_directory = None
_figures_prefix = ""


def configure_figure_export(directory):
    """!
    @brief Activa el guardado en PNG de cada figura que se muestre en el notebook.
    @param directory Carpeta de destino; se crea si no existe y se eliminan sus PNG previos.
    @return Ruta absoluta de la carpeta de destino.
    """
    global _figures_directory
    # Se parte del estilo por omisión para que el tema del IDE (p. ej. texto blanco del
    # modo oscuro de PyCharm) no llegue a las figuras del informe.
    matplotlib.rcdefaults()
    _figures_directory = Path(directory).resolve()
    _figures_directory.mkdir(parents=True, exist_ok=True)
    for file in _figures_directory.glob("*.png"):
        file.unlink()
    return _figures_directory


def _figure_filename(figure):
    """!
    @brief Construye un nombre de archivo estable a partir del título de la figura.
    @param figure Figura de Matplotlib que se va a guardar.
    @return Nombre sin extensión, en minúsculas, ASCII y separado por guiones.
    """
    title_text = figure._suptitle.get_text() if figure._suptitle is not None else ""
    if not title_text:
        title_text = next((axis.get_title() for axis in figure.axes if axis.get_title()), "figura")
    title_text = f"{_figures_prefix} {title_text.splitlines()[0]}".replace("–", "-")
    title_text = unicodedata.normalize("NFKD", title_text).encode("ascii", "ignore").decode()
    base = re.sub(r"[^a-z0-9]+", "-", re.sub(r"[$_{}\\]", "", title_text.lower())).strip("-")

    return base


def _show_figure(figure):
    """!
    @brief Guarda la figura si la exportación está activa y luego la muestra.
    @param figure Figura de Matplotlib ya terminada.
    @return None.
    """
    if _figures_directory is not None:
        output_path = _figures_directory / f"{_figure_filename(figure)}.png"
        figure.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()


def _points_as_tensor(points):
    """!
    @brief Convierte puntos bidimensionales en un tensor de forma (n, 2).
    @param points Punto o colección de puntos que se desea normalizar.
    @return Tensor en CPU con forma (n, 2), o None si no hay puntos.
    """
    if points is None:
        return None

    if isinstance(points, torch.Tensor):
        points_tensor = points.detach().cpu()
    else:
        points_list = list(points)
        if len(points_list) == 0:
            return None
        points_tensor = torch.stack(
            [torch.as_tensor(point).detach().cpu() for point in points_list]
        )

    if points_tensor.ndim == 1:
        points_tensor = points_tensor.unsqueeze(0)

    if points_tensor.ndim != 2 or points_tensor.shape[1] != 2:
        raise ValueError("Los puntos deben tener forma (n, 2).")
    if not torch.isfinite(points_tensor).all():
        raise ValueError("El historial contiene NaN o infinito; GD pudo divergir.")

    return points_tensor


def _plot_minima(axis, minima, local_minima):
    """!
    @brief Dibuja los mínimos globales y locales conocidos sobre unos ejes.
    @param axis Eje de Matplotlib donde se agregan los marcadores.
    @param minima Uno o varios mínimos globales conocidos.
    @param local_minima Mínimos locales que se muestran con marcadores pequeños.
    @return None.
    """
    minima_tensor = _points_as_tensor(minima)
    if minima_tensor is not None:
        minima_values = minima_tensor.numpy()
        label = "Mínimo global" if len(minima_values) == 1 else "Mínimos globales"
        axis.scatter(
            minima_values[:, 0],
            minima_values[:, 1],
            marker="X",
            s=80,
            color="gold",
            edgecolor="black",
            label=label,
            zorder=5,
        )

    local_minima_tensor = _points_as_tensor(local_minima)
    if local_minima_tensor is not None:
        local_minima_values = local_minima_tensor.numpy()
        axis.scatter(
            local_minima_values[:, 0],
            local_minima_values[:, 1],
            marker="o",
            s=12,
            color="gold",
            edgecolor="black",
            linewidth=0.25,
            label="Mínimos locales",
            zorder=5,
        )


def _title_with_hyperparameters(title, hyperparameters):
    """!
    @brief Agrega al título una línea con los hiperparámetros usados.
    @param title Título principal de la gráfica.
    @param hyperparameters Diccionario opcional con nombre y valor de cada hiperparámetro.
    @return Título con el subtítulo de hiperparámetros, si se proporcionaron.
    """
    if not hyperparameters:
        return title

    # Los enteros, como la cantidad de partículas, se muestran sin decimales.
    formatted_parameters = []
    for name, scalar_value in hyperparameters.items():
        display_name = "partículas" if name == "n_particles" else name
        formatted_parameters.append(
            f"{display_name}={scalar_value}"
            if isinstance(scalar_value, numbers.Integral)
            else f"{display_name}={scalar_value:.5f}"
        )
    hyperparameter_text = ", ".join(formatted_parameters)
    return f"{title}\n{hyperparameter_text}"


def _particle_style(particle_index):
    """!
    @brief Devuelve el color, el trazo y el marcador asignados a una partícula.
    @param particle_index Índice de la partícula dentro del enjambre.
    @return Tupla con color, estilo de línea y marcador.
    """
    # Después de diez partículas los colores se repiten y cambian el trazo y el marcador;
    # los cinco estilos distinguen hasta 50 partículas. La paleta fija evita que el estilo
    # activo de Matplotlib (Optuna aplica ggplot) repita colores antes de la décima.
    styles = [
        ("--", "o"),
        ("-.", "s"),
        (":", "^"),
        ("-", "D"),
        ((0, (5, 1, 1, 1, 1, 1)), "v"),
    ]
    color = matplotlib.colormaps["tab10"].colors[particle_index % 10]
    line_style, marker_style = styles[(particle_index // 10) % len(styles)]
    return color, line_style, marker_style


def _outside_legend(figure, axis, max_rows=25):
    """!
    @brief Coloca la leyenda a la derecha de la figura y la reparte en columnas si es larga.
    @param figure Figura de Matplotlib que recibe la leyenda.
    @param axis Eje cuyos elementos con etiqueta forman la leyenda.
    @param max_rows Filas por columna antes de agregar otra columna.
    @return None.
    """
    entry_count = len(axis.get_legend_handles_labels()[1])
    columns = max(1, math.ceil(entry_count / max_rows))
    figure.legend(
        loc="outside right upper",
        ncols=columns,
        fontsize="small" if columns > 1 else "medium",
    )


def plot_critical_points(
    x_grid,
    y_grid,
    values,
    global_minima,
    local_minima,
    saddle_points,
    title,
    view_limits=None,
):
    """!
    @brief Grafica mínimos y puntos silla sobre las curvas de nivel.
    @param x_grid Malla con las coordenadas x.
    @param y_grid Malla con las coordenadas y.
    @param values Valores de la función sobre la malla.
    @param global_minima Lista de tensores con mínimos globales.
    @param local_minima Lista de tensores con mínimos locales no globales.
    @param saddle_points Lista de tensores con puntos silla.
    @param title Título de la gráfica.
    @param view_limits Límites opcionales (mínimo, máximo) para ambos ejes.
    @return None.
    """
    x_values = x_grid.detach().cpu().numpy()
    y_values = y_grid.detach().cpu().numpy()

    # El logaritmo solo cambia la escala de color y permite ver mejor los valles.
    color_values = torch.log1p(values.clamp_min(0)).detach().cpu().numpy()

    figure, axis = plt.subplots(figsize=(7, 6))
    contours = axis.contourf(
        x_values,
        y_values,
        color_values,
        levels=50,
        cmap="viridis",
    )
    colorbar = figure.colorbar(contours, ax=axis)
    colorbar.set_label("log(1 + f(x, y))")

    if global_minima:
        points = torch.stack(global_minima).detach().cpu().numpy()
        axis.scatter(
            points[:, 0],
            points[:, 1],
            marker="*",
            s=220,
            color="gold",
            edgecolor="black",
            label="Mínimo global",
            zorder=4,
        )
        for x_coordinate, y_coordinate in points:
            text_offset = (-6, 6) if x_coordinate > 0 else (6, 6)
            horizontal_alignment = "right" if x_coordinate > 0 else "left"
            axis.annotate(
                f"({x_coordinate:.3f}, {y_coordinate:.3f})",
                (x_coordinate, y_coordinate),
                xytext=text_offset,
                textcoords="offset points",
                fontsize=8,
                ha=horizontal_alignment,
            )

    if local_minima:
        points = torch.stack(local_minima).detach().cpu().numpy()
        axis.scatter(
            points[:, 0],
            points[:, 1],
            marker="o",
            s=45,
            color="deepskyblue",
            edgecolor="black",
            label="Mínimo local",
            zorder=3,
        )

    if saddle_points:
        points = torch.stack(saddle_points).detach().cpu().numpy()
        axis.scatter(
            points[:, 0],
            points[:, 1],
            marker="x",
            s=65,
            color="crimson",
            label="Punto silla",
            zorder=3,
        )

    if view_limits is not None:
        axis.set_xlim(view_limits)
        axis.set_ylim(view_limits)

    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_aspect("equal", adjustable="box")
    axis.legend()
    _show_figure(figure)


def plot_learning_curve(function_values, title="Curva de aprendizaje"):
    """!
    @brief Grafica el valor de la función después de cada paso del algoritmo.
    @param function_values Historial de valores f(x_t).
    @param title Título de la gráfica.
    @return La figura y sus ejes para permitir ajustes posteriores.
    """
    values = torch.as_tensor(function_values).detach().cpu().flatten()
    if values.numel() == 0:
        raise ValueError("El historial de valores no puede estar vacío.")
    if not torch.isfinite(values).all():
        raise ValueError("El historial contiene NaN o infinito; GD pudo divergir.")

    iterations = torch.arange(values.numel())
    plot_values = values.numpy()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        iterations,
        values,
        color="tab:blue",
        linewidth=2,
        label=r"$f(x_t)$",
    )
    axis.scatter(
        iterations[0],
        values[0],
        marker="s",
        s=75,
        color="forestgreen",
        edgecolor="black",
        label="Inicio",
        zorder=3,
    )
    axis.scatter(
        iterations[-1],
        values[-1],
        marker="*",
        s=150,
        color="crimson",
        edgecolor="black",
        label="Final",
        zorder=3,
    )

    # La escala lineal aplana el tramo final cuando las primeras iteraciones
    # son varios órdenes de magnitud mayores que las últimas.
    absolute_values = np.abs(plot_values[plot_values != 0])
    if absolute_values.size and absolute_values.max() / absolute_values.min() >= 100:
        if np.all(plot_values > 0):
            axis.set_yscale("log")
        else:
            axis.set_yscale("symlog", linthresh=max(float(absolute_values.min()), 1e-8))
        axis.set_ylabel(r"$f(x_t)$ (escala logarítmica)")
    else:
        axis.set_ylabel(r"$f(x_t)$")

    axis.set_title(title)
    axis.set_xlabel("Iteración t")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    _show_figure(figure)
    return figure, axis


def plot_func_hist(
    x_grid,
    y_grid,
    func,
    visited_points,
    title,
    minima=None,
    local_minima=None,
    view_limits=None,
):
    """!
    @brief Grafica la trayectoria sobre una malla que cubre todos los puntos visitados.
    @param x_grid Valores del eje x usados para construir la malla.
    @param y_grid Valores del eje y usados para construir la malla.
    @param func Función que recibe un tensor con las coordenadas x e y.
    @param visited_points Puntos recorridos por el algoritmo, con forma (n, 2).
    @param title Título de la gráfica.
    @param minima Uno o varios mínimos globales conocidos.
    @param local_minima Mínimos locales que se muestran con marcadores pequeños.
    @param view_limits Límites opcionales (mínimo, máximo) que sustituyen la vista automática.
    @return La figura y sus ejes para permitir ajustes posteriores.
    """
    x_values = torch.as_tensor(x_grid).detach().cpu()
    y_values = torch.as_tensor(y_grid).detach().cpu()
    path = _points_as_tensor(visited_points)
    if path is None or path.shape[0] == 0:
        raise ValueError("El historial de puntos no puede estar vacío.")

    original_x_limits = (float(x_values.min()), float(x_values.max()))
    original_y_limits = (float(y_values.min()), float(y_values.max()))
    original_x_values = x_values
    original_y_values = y_values
    visible_points = [path.numpy()]
    for known_points in (minima, local_minima):
        point_tensor = _points_as_tensor(known_points)
        if point_tensor is not None:
            visible_points.append(point_tensor.numpy())
    visible_points = np.concatenate(visible_points)

    x_lower_bound = min(original_x_limits[0], float(visible_points[:, 0].min()))
    x_upper_bound = max(original_x_limits[1], float(visible_points[:, 0].max()))
    y_lower_bound = min(original_y_limits[0], float(visible_points[:, 1].min()))
    y_upper_bound = max(original_y_limits[1], float(visible_points[:, 1].max()))
    expanded_grid = view_limits is None and (
        x_lower_bound < original_x_limits[0]
        or x_upper_bound > original_x_limits[1]
        or y_lower_bound < original_y_limits[0]
        or y_upper_bound > original_y_limits[1]
    )
    if expanded_grid:
        # El margen depende del recorrido; se conserva la cantidad original de muestras.
        x_padding = 0.05 * (x_upper_bound - x_lower_bound)
        y_padding = 0.05 * (y_upper_bound - y_lower_bound)
        x_values = torch.linspace(
            x_lower_bound - x_padding,
            x_upper_bound + x_padding,
            x_values.numel(),
            dtype=x_values.dtype,
        )
        y_values = torch.linspace(
            y_lower_bound - y_padding,
            y_upper_bound + y_padding,
            y_values.numel(),
            dtype=y_values.dtype,
        )

    X1, X2 = torch.meshgrid(x_values, y_values, indexing="ij")
    with torch.no_grad():
        values = func(torch.stack((X1, X2)))
        if expanded_grid:
            X1_original, X2_original = torch.meshgrid(
                original_x_values, original_y_values, indexing="ij"
            )
            original_values = func(torch.stack((X1_original, X2_original)))

    # El cambio de escala hace visibles los valles sin modificar la trayectoria.
    minimum_value = values.min()
    levels = 50
    extension = "neither"
    if expanded_grid:
        minimum_value = torch.minimum(minimum_value, original_values.min())
        # La escala conserva el contraste del dominio de referencia aunque
        # aparezcan valores mayores en la zona nueva.
        reference_max = torch.log1p(
            (original_values - minimum_value).clamp_min(0)
        ).max().item()
        if reference_max > 0:
            levels = np.linspace(0, reference_max, 51)
            extension = "max"
    color_values = torch.log1p((values - minimum_value).clamp_min(0))

    figure, axis = plt.subplots(figsize=(8, 6.5))
    contours = axis.contourf(
        X1.numpy(),
        X2.numpy(),
        color_values.detach().cpu().numpy(),
        levels=levels,
        cmap="viridis",
        extend=extension,
    )
    colorbar = figure.colorbar(contours, ax=axis)
    colorbar.set_label(r"$\log(1 + f(x,y) - f_{min})$")

    if expanded_grid:
        axis.add_patch(
            Rectangle(
                (original_x_limits[0], original_y_limits[0]),
                original_x_limits[1] - original_x_limits[0],
                original_y_limits[1] - original_y_limits[0],
                fill=False,
                edgecolor="black",
                linestyle="--",
                linewidth=1.2,
                label="Dominio de referencia",
                zorder=3,
            )
        )

    path_values = path.numpy()
    marker_interval = max(1, int(np.ceil(len(path_values) / 15)))
    trajectory_line, = axis.plot(
        path_values[:, 0],
        path_values[:, 1],
        color="white",
        linewidth=2,
        marker="o",
        markersize=3,
        markevery=marker_interval,
        markerfacecolor="white",
        markeredgecolor="black",
        label="Trayectoria",
        zorder=4,
    )
    trajectory_line.set_path_effects(
        [patheffects.Stroke(linewidth=2.8, foreground="black"), patheffects.Normal()]
    )
    axis.scatter(
        path_values[0, 0],
        path_values[0, 1],
        marker="s",
        s=95,
        color="forestgreen",
        edgecolor="black",
        label="Inicio",
        zorder=5,
    )
    axis.scatter(
        path_values[-1, 0],
        path_values[-1, 1],
        marker="*",
        s=120,
        color="crimson",
        edgecolor="black",
        label="Final",
        zorder=7,
    )

    _plot_minima(axis, minima, local_minima)

    if view_limits is not None:
        axis.set_xlim(view_limits)
        axis.set_ylim(view_limits)
    else:
        axis.set_xlim(float(x_values.min()), float(x_values.max()))
        axis.set_ylim(float(y_values.min()), float(y_values.max()))

    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    figure.tight_layout()
    _show_figure(figure)
    return figure, axis


def plot_pso_history(
    func,
    swarm_history,
    title,
    minima,
    tolerance=1e-3,
    hyperparameters=None,
    particles=None,
    highlight_minimum=False,
):
    """!
    @brief Grafica en escala lineal las partículas de PSO seleccionadas.
    @param func Función que recibe un tensor con las coordenadas x e y.
    @param swarm_history Posiciones de todas las partículas, con forma (T+1, n, 2).
    @param title Título de la gráfica.
    @param minima Uno o varios mínimos globales calculados para la función.
    @param tolerance Brecha máxima respecto al mínimo para considerar convergencia.
    @param hyperparameters Diccionario opcional con los hiperparámetros del subtítulo.
    @param particles Índices opcionales de las partículas graficadas; por omisión, todas.
    @param highlight_minimum Si es True, marca el menor valor de las partículas graficadas.
    @return La figura y sus ejes para permitir ajustes posteriores.
    """
    swarm = torch.as_tensor(swarm_history).detach().cpu()
    if swarm.ndim != 3 or swarm.shape[2] != 2:
        raise ValueError("El historial del enjambre debe tener forma (T+1, n, 2).")
    minima_tensor = _points_as_tensor(minima)
    if minima_tensor is None:
        raise ValueError("Se necesita al menos un mínimo global de la función.")

    # El mínimo global se evalúa en los mínimos calculados, no se asume igual a 0.
    with torch.no_grad():
        numeric_values = func(swarm.permute(2, 0, 1))
        minimum_value = func(minima_tensor.T).min().item()
    if not torch.isfinite(numeric_values).all():
        raise ValueError("El historial contiene NaN o infinito; PSO pudo divergir.")

    numeric_values = numeric_values.numpy()
    positions = swarm.numpy()
    iteration_count, n_particles = numeric_values.shape
    indices = list(range(n_particles)) if particles is None else list(particles)
    iterations = np.arange(iteration_count)
    convergence_value = minimum_value + tolerance

    figure, axis = plt.subplots(
        figsize=(13, 6) if len(indices) > 1 else (12, 6),
        layout="constrained",
    )
    axis.axhline(
        convergence_value,
        color="darkred",
        linestyle=":",
        linewidth=1.5,
        label=f"Convergencia = {convergence_value:g}",
        zorder=4,
    )

    # Cada partícula conserva su color y trazo aunque solo se grafique una parte del enjambre.
    marker_interval = max(1, iteration_count // 12)
    for particle_index in indices:
        color, line_style, marker_style = _particle_style(particle_index)
        x_0, y_0 = positions[0, particle_index]
        axis.plot(
            iterations,
            numeric_values[:, particle_index],
            color=color,
            linestyle=line_style,
            marker=marker_style,
            markevery=marker_interval,
            markersize=3,
            linewidth=1.3,
            alpha=0.8,
            label=f"Partícula {particle_index + 1}: ({x_0:.2f}, {y_0:.2f})",
        )

    if highlight_minimum:
        plotted_values = numeric_values[:, indices]
        iteration, column_name = np.unravel_index(
            np.argmin(plotted_values),
            plotted_values.shape,
        )
        highlighted_value = plotted_values[iteration, column_name]
        x_min, y_min = positions[iteration, indices[column_name]]
        axis.scatter(
            iteration,
            highlighted_value,
            marker="*",
            s=320,
            color="gold",
            edgecolor="black",
            label=(
                f"Más cercano al mínimo: f = {highlighted_value:.4g} en t = {iteration}\n"
                f"(x, y) = ({x_min:.4f}, {y_min:.4f})"
            ),
            zorder=5,
        )

    axis.set_title(_title_with_hyperparameters(title, hyperparameters))
    axis.set_xlabel("Iteración")
    axis.set_ylabel(r"$f(x, y)$")
    axis.grid(alpha=0.3)
    _outside_legend(figure, axis)
    _show_figure(figure)
    return figure, axis


def plot_pso_history_groups(
    func,
    swarm_history,
    title,
    minima,
    tolerance=1e-3,
    hyperparameters=None,
):
    """!
    @brief Grafica el enjambre de PSO en grupos consecutivos de hasta diez partículas.
    @param func Función que recibe un tensor con las coordenadas x e y.
    @param swarm_history Posiciones de todas las partículas, con forma (T+1, n, 2).
    @param title Título común de las gráficas.
    @param minima Uno o varios mínimos globales calculados para la función.
    @param tolerance Brecha máxima respecto al mínimo para considerar convergencia.
    @param hyperparameters Diccionario opcional con los hiperparámetros del subtítulo.
    @return Lista de pares (figura, ejes), uno por cada grupo de partículas.
    """
    swarm = torch.as_tensor(swarm_history)
    if swarm.ndim != 3 or swarm.shape[2] != 2 or swarm.shape[1] == 0:
        raise ValueError("El historial del enjambre debe tener forma (T+1, n, 2) con n > 0.")

    plots = []
    for start in range(0, swarm.shape[1], 10):
        end = min(start + 10, swarm.shape[1])
        plots.append(
            plot_pso_history(
                func,
                swarm,
                title=f"{title} (partículas {start + 1}–{end})",
                minima=minima,
                tolerance=tolerance,
                hyperparameters=hyperparameters,
                particles=range(start, end),
            )
        )
    return plots


def best_pso_particle(func, swarm_history):
    """!
    @brief Encuentra la partícula que llegó al valor más cercano al mínimo global.
    @param func Función que recibe un tensor con las coordenadas x e y.
    @param swarm_history Posiciones de todas las partículas, con forma (T+1, n, 2).
    @return Tupla con el índice de la partícula, la iteración y el menor valor alcanzado.
    """
    swarm = torch.as_tensor(swarm_history).detach().cpu()
    if swarm.ndim != 3 or swarm.shape[2] != 2:
        raise ValueError("El historial del enjambre debe tener forma (T+1, n, 2).")

    with torch.no_grad():
        numeric_values = func(swarm.permute(2, 0, 1))
    if not torch.isfinite(numeric_values).all():
        raise ValueError("El historial contiene NaN o infinito; PSO pudo divergir.")

    # Como f nunca es menor que su mínimo global, el menor valor es el más cercano a él.
    iteration, particle_index = divmod(int(torch.argmin(numeric_values)), numeric_values.shape[1])
    return particle_index, iteration, float(numeric_values[iteration, particle_index])


def plot_best_pso_particle(
    func,
    swarm_history,
    title,
    minima,
    tolerance=1e-3,
    hyperparameters=None,
):
    """!
    @brief Grafica la evolución de f en la partícula que llegó más cerca del mínimo global.
    @param func Función que recibe un tensor con las coordenadas x e y.
    @param swarm_history Posiciones de todas las partículas, con forma (T+1, n, 2).
    @param title Título de la gráfica.
    @param minima Uno o varios mínimos globales calculados para la función.
    @param tolerance Brecha máxima respecto al mínimo para considerar convergencia.
    @param hyperparameters Diccionario opcional con los hiperparámetros del subtítulo.
    @return La figura y sus ejes para permitir ajustes posteriores.
    """
    particle_index, _, _ = best_pso_particle(func, swarm_history)
    return plot_pso_history(
        func,
        swarm_history,
        title=f"{title} (partícula {particle_index + 1})",
        minima=minima,
        tolerance=tolerance,
        hyperparameters=hyperparameters,
        particles=[particle_index],
        highlight_minimum=True,
    )


def plot_learning_curves(
    value_histories,
    curve_labels,
    title,
    minimum_value,
    tolerance=1e-3,
    hyperparameters=None,
):
    """!
    @brief Grafica varias curvas de aprendizaje y su promedio en escala lineal.
    @param value_histories Historiales f(x_t) de igual longitud, uno por corrida.
    @param curve_labels Texto de la leyenda para cada historial.
    @param title Título de la gráfica.
    @param minimum_value Valor mínimo global conocido de la función.
    @param tolerance Brecha máxima respecto al mínimo para considerar convergencia.
    @param hyperparameters Diccionario opcional con los hiperparámetros del subtítulo.
    @return La figura y sus ejes para permitir ajustes posteriores.
    """
    curves = torch.stack(
        [torch.as_tensor(history).detach().cpu().flatten() for history in value_histories]
    )
    if curves.shape[0] != len(curve_labels):
        raise ValueError("Se necesita una etiqueta por cada curva de aprendizaje.")
    if not torch.isfinite(curves).all():
        raise ValueError("Los historiales contienen NaN o infinito.")

    curves = curves.numpy()
    iterations = np.arange(curves.shape[1])
    convergence_value = minimum_value + tolerance

    figure, axis = plt.subplots(figsize=(13, 6), layout="constrained")
    axis.axhline(
        convergence_value,
        color="darkred",
        linestyle=":",
        linewidth=1.5,
        label=f"Convergencia = {convergence_value:g}",
        zorder=4,
    )
    marker_interval = max(1, len(iterations) // 12)
    for index, (curve, curve_label) in enumerate(zip(curves, curve_labels)):
        axis.plot(
            iterations,
            curve,
            color=f"C{index % 10}",
            linestyle="--",
            marker="o",
            markevery=marker_interval,
            markersize=3,
            linewidth=1.2,
            alpha=0.75,
            label=curve_label,
        )
    axis.plot(
        iterations,
        curves.mean(axis=0),
        color="black",
        linewidth=2.5,
        label="Promedio",
    )

    axis.set_title(_title_with_hyperparameters(title, hyperparameters))
    axis.set_xlabel("Iteración")
    axis.set_ylabel(r"$f(x_t)$")
    axis.grid(alpha=0.3)
    figure.legend(loc="outside right upper")
    _show_figure(figure)
    return figure, axis


def display_table(table, title_text, formats=None, na_rep="—"):
    """!
    @brief Muestra un DataFrame con un título visible y formato consistente.
    @param table DataFrame de pandas que se desea mostrar.
    @param title_text Título que aparecerá como encabezado de la tabla.
    @param formats Mapeo opcional de columnas a formatos de pandas Styler.
    @param na_rep Texto usado para representar valores ausentes.
    @return None.
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError("La tabla debe ser un DataFrame de pandas.")

    table_id = hashlib.sha256(title_text.encode("utf-8")).hexdigest()[:12]
    style = table.style.set_uuid(table_id).set_caption(title_text).set_table_styles(
        [
            {
                "selector": "caption",
                "props": [
                    ("caption-side", "top"),
                    ("font-size", "1.1em"),
                    ("font-weight", "bold"),
                    ("text-align", "left"),
                    ("padding-bottom", "0.5em"),
                ],
            }
        ],
        overwrite=False,
    )
    style = style.format(formats, na_rep=na_rep)
    display(style)


def plot_function_2d_3d(x_grid, y_grid, values, title, z_label):
    """!
    @brief Grafica curvas de nivel y una superficie 3D de una función muestreada.
    @param x_grid Malla con las coordenadas x.
    @param y_grid Malla con las coordenadas y.
    @param values Valores de la función sobre la malla.
    @param title Título descriptivo de la figura.
    @param z_label Etiqueta del valor de la función.
    @return None.
    """
    x_values = torch.as_tensor(x_grid).detach().cpu().numpy()
    y_values = torch.as_tensor(y_grid).detach().cpu().numpy()
    z_values = torch.as_tensor(values).detach().cpu().numpy()
    color_values = np.log1p(np.clip(z_values - z_values.min(), 0, None))
    x_limits = (float(x_values.min()), float(x_values.max()))
    y_limits = (float(y_values.min()), float(y_values.max()))

    figure = plt.figure(figsize=(14, 5), constrained_layout=True)
    figure.suptitle(title, fontsize=14)

    contour_axis = figure.add_subplot(1, 2, 1)
    contours = contour_axis.contourf(
        x_values,
        y_values,
        color_values,
        levels=50,
        cmap="viridis",
    )
    contour_axis.set(
        xlabel="x",
        ylabel="y",
        title="Curvas de nivel",
        xlim=x_limits,
        ylim=y_limits,
    )
    contour_axis.set_aspect("equal", adjustable="box")
    contour_colorbar = figure.colorbar(contours, ax=contour_axis)
    contour_colorbar.set_label(r"$\log(1 + f(x,y) - f_{\min})$")

    surface_axis = figure.add_subplot(1, 2, 2, projection="3d")
    surface = surface_axis.plot_surface(
        x_values,
        y_values,
        z_values,
        cmap="viridis",
        rstride=2,
        cstride=2,
        linewidth=0,
        antialiased=True,
    )
    surface_axis.set(
        xlabel="x",
        ylabel="y",
        zlabel=z_label,
        title="Superficie 3D",
        xlim=x_limits,
        ylim=y_limits,
    )
    surface_axis.view_init(elev=28, azim=-55)
    surface_axis.set_box_aspect((1, 1, 0.65))
    surface_colorbar = figure.colorbar(
        surface,
        ax=surface_axis,
        shrink=0.72,
        pad=0.10,
    )
    surface_colorbar.set_label(z_label)
    _show_figure(figure)


def grid_minimum_coordinates(x_grid, y_grid, values):
    """!
    @brief Obtiene las coordenadas del menor valor guardado en una malla.
    @param x_grid Malla con las coordenadas x.
    @param y_grid Malla con las coordenadas y.
    @param values Valores de la función sobre la malla.
    @return Tensor con las coordenadas aproximadas del mínimo.
    """
    flat_index = torch.argmin(values)
    row = flat_index // values.shape[1]
    column = flat_index % values.shape[1]
    return torch.stack((x_grid[row, column], y_grid[row, column]))


def gradient_and_hessian(function, point):
    """!
    @brief Calcula valor, gradiente y Hessiana de una función en un punto.
    @param function Función de dos variables implementada con PyTorch.
    @param point Tensor de dos componentes con requires_grad=True.
    @return Tupla con valor, gradiente y matriz Hessiana.
    """
    scalar_value = function(point)
    gradient = torch.autograd.grad(scalar_value, point, create_graph=True)[0]
    hessian_rows = [
        torch.autograd.grad(component, point, retain_graph=True)[0] for component in gradient
    ]
    return scalar_value, gradient, torch.stack(hessian_rows)


def find_stationary_point(
    function,
    start,
    max_iterations=20,
    tolerance=1e-8,
):
    """!
    @brief Refina un punto inicial hasta que su gradiente sea cercano a cero.
    @param function Función de dos variables implementada con PyTorch.
    @param start Tupla con las coordenadas iniciales.
    @param max_iterations Máximo de pasos de Newton.
    @param tolerance Norma máxima del gradiente para detener el método.
    @return Tensor con el punto estacionario aproximado.
    """
    point = torch.tensor(start, dtype=torch.float64, requires_grad=True)

    for _ in range(max_iterations):
        _, gradient, hessian = gradient_and_hessian(function, point)
        if torch.linalg.vector_norm(gradient) < tolerance:
            break

        step = torch.linalg.solve(hessian, gradient)
        with torch.no_grad():
            point -= step

    return point.detach()


def classify_point(function, point, tolerance=1e-6):
    """!
    @brief Clasifica un punto mediante el gradiente y la Hessiana 2x2.
    @param function Función de dos variables implementada con PyTorch.
    @param point Tensor con las coordenadas del candidato.
    @param tolerance Tolerancia usada en las comparaciones numéricas.
    @return Diccionario con coordenadas, valor y comprobaciones.
    """
    autodiff_point = point.clone().detach().requires_grad_(True)
    scalar_value, gradient, hessian = gradient_and_hessian(function, autodiff_point)
    gradient_norm = torch.linalg.vector_norm(gradient).item()
    determinant = torch.det(hessian).item()
    f_xx = hessian[0, 0].item()

    if gradient_norm > tolerance:
        kind = "no convergió"
    elif determinant < -tolerance:
        kind = "punto silla"
    elif determinant > tolerance and f_xx > 0:
        kind = "mínimo local"
    elif determinant > tolerance and f_xx < 0:
        kind = "máximo local"
    else:
        kind = "prueba inconclusa"

    return {
        "point": point.detach(),
        "tipo": kind,
        "x": point[0].item(),
        "y": point[1].item(),
        "valor": scalar_value.item(),
        "norma_gradiente": gradient_norm,
        "det_hessiana": determinant,
    }


def results_to_table(results):
    """!
    @brief Convierte resultados de puntos críticos en una tabla compacta.
    @param results Lista de diccionarios creados por classify_point.
    @return DataFrame con los valores necesarios para verificar cada punto.
    """
    columns = [
        "tipo",
        "x",
        "y",
        "valor",
        "norma_gradiente",
        "det_hessiana",
    ]
    rows = [{column_name: result[column_name] for column_name in columns} for result in results]
    table = pd.DataFrame(rows)
    table["norma_gradiente"] = table["norma_gradiente"].map(lambda value: f"{value:.2e}")
    return table.round(
        {
            "x": 6,
            "y": 6,
            "valor": 6,
            "det_hessiana": 6,
        }
    )


def find_ackley_minima_in_domain(
    function,
    lower_bound,
    upper_bound,
):
    """!
    @brief Busca mínimos locales de Ackley desde una cuadrícula de enteros.
    @param function Implementación de la función de Ackley.
    @param lower_bound Extremo inferior del dominio en ambos ejes.
    @param upper_bound Extremo superior del dominio en ambos ejes.
    @return Lista sin duplicados de mínimos locales dentro del dominio.
    """
    minimum_points = []
    for x0 in range(math.ceil(lower_bound), math.floor(upper_bound) + 1):
        for y0 in range(math.ceil(lower_bound), math.floor(upper_bound) + 1):
            if x0 == 0 and y0 == 0:
                continue

            point = find_stationary_point(
                function,
                start=(float(x0), float(y0)),
            )
            result = classify_point(function, point)
            inside_domain = torch.all(
                (point >= lower_bound) & (point <= upper_bound)
            ).item()
            duplicate = any(
                torch.linalg.vector_norm(point - previous).item() < 1e-5 for previous in minimum_points
            )

            if result["tipo"] == "mínimo local" and inside_domain and not duplicate:
                minimum_points.append(point)

    return minimum_points


def value_and_gradient(function, point):
    """!
    @brief Calcula el valor de una función y su gradiente en un punto.
    @param function Función objetivo implementada con PyTorch.
    @param point Tensor con respecto al cual se calcula el gradiente.
    @return Tupla con el valor de la función y su gradiente.
    """
    scalar_value = function(point)
    gradient = torch.autograd.grad(scalar_value, point)[0]
    return scalar_value, gradient


def generate_initial_point(lower_bound, upper_bound, generator=None):
    """!
    @brief Genera un punto uniforme al azar dentro de un dominio cuadrado.
    @param lower_bound Extremo inferior del dominio.
    @param upper_bound Extremo superior del dominio.
    @param generator Generador opcional compatible con random.Random.
    @return Tupla con dos coordenadas aleatorias.
    """
    source_rng = generator if generator is not None else random
    return (
        source_rng.uniform(lower_bound, upper_bound),
        source_rng.uniform(lower_bound, upper_bound),
    )


def evaluate_convergence(value_history, minimum_value, tolerance):
    """!
    @brief Localiza la primera iteración que satisface la tolerancia objetivo.
    @param value_history Valores de la función para cada punto visitado.
    @param minimum_value Valor mínimo global conocido de la función.
    @param tolerance Brecha máxima permitida respecto al mínimo.
    @return Diccionario con estado, iteración, índice, valor y brecha reportados.
    """
    for index, scalar_value in enumerate(value_history):
        numeric_value = float(scalar_value.detach().cpu())
        if not math.isfinite(numeric_value):
            return {
                "converged": False,
                "diverged": True,
                "iterations": None,
                "report_index": index,
                "reported_value": math.nan,
                "gap": math.inf,
            }

        gap = abs(numeric_value - minimum_value)
        if gap <= tolerance:
            return {
                "converged": True,
                "diverged": False,
                "iterations": index,
                "report_index": index,
                "reported_value": numeric_value,
                "gap": gap,
            }

    final_value = float(value_history[-1].detach().cpu())
    return {
        "converged": False,
        "diverged": not math.isfinite(final_value),
        "iterations": None,
        "report_index": len(value_history) - 1,
        "reported_value": final_value,
        "gap": abs(final_value - minimum_value),
    }


def create_optuna_objective(
    algorithm,
    configuration,
    initial_points,
    iterations,
    tolerance,
    epsilon_rmsprop,
    run_gd_fn,
    run_rmsprop_fn,
    run_pso_fn=None,
    pso_seed=0,
):
    """!
    @brief Crea un objetivo Optuna basado en el promedio del valor final.
    @param algorithm Nombre del algoritmo: GD, RMSProp o PSO.
    @param configuration Función, mínimo conocido y rangos de búsqueda.
    @param initial_points Puntos compartidos por todos los ensayos.
    @param iterations Cantidad fija de actualizaciones por corrida.
    @param tolerance Tolerancia usada solo para métricas diagnósticas.
    @param epsilon_rmsprop Constante de estabilidad fija de RMSProp.
    @param run_gd_fn Función que ejecuta descenso del gradiente.
    @param run_rmsprop_fn Función que ejecuta RMSProp.
    @param run_pso_fn Función que ejecuta el enjambre de partículas.
    @param pso_seed Semilla base de PSO; cada punto usa semilla_pso + su índice.
    @return Función objetivo compatible con Optuna.
    """
    if algorithm == "PSO" and run_pso_fn is None:
        raise ValueError("Se necesita ejecutar_pso para calibrar PSO.")

    function = configuration["function"]
    minimum_value = configuration["minimum_value"]

    def objective(trial):
        """!
        @brief Evalúa un ensayo con el presupuesto fijo del estudio.
        @param trial Ensayo de Optuna que contiene los hiperparámetros sugeridos.
        @return Promedio del valor final de la función en los puntos iniciales.
        """
        if algorithm == "GD":
            alpha = trial.suggest_float(
                "alpha",
                configuration["gd_alpha_min"],
                configuration["gd_alpha_max"],
                log=True,
            )
            gamma = None
        elif algorithm == "RMSProp":
            alpha = trial.suggest_float(
                "alpha",
                configuration["rms_alpha_min"],
                configuration["rms_alpha_max"],
                log=True,
            )
            gamma = trial.suggest_float(
                "gamma",
                configuration["gamma_min"],
                configuration["gamma_max"],
            )
        elif algorithm == "PSO":
            c1 = trial.suggest_float(
                "c1",
                configuration["pso_c1_min"],
                configuration["pso_c1_max"],
            )
            c2 = trial.suggest_float(
                "c2",
                configuration["pso_c2_min"],
                configuration["pso_c2_max"],
            )
            n_particles = trial.suggest_int(
                "n_particles",
                configuration["pso_particles_min"],
                configuration["pso_particles_max"],
            )
        else:
            raise ValueError(f"Algoritmo no soportado: {algorithm}")

        results = []
        final_values = []

        for point_index, initial_point in enumerate(initial_points):
            try:
                if algorithm == "GD":
                    _, value_history = run_gd_fn(
                        alpha=alpha,
                        t=iterations,
                        func=function,
                        initial_point=initial_point,
                    )
                elif algorithm == "RMSProp":
                    _, value_history = run_rmsprop_fn(
                        alpha=alpha,
                        gamma=gamma,
                        epsilon=epsilon_rmsprop,
                        t=iterations,
                        func=function,
                        initial_point=initial_point,
                    )
                else:
                    # La semilla por punto hace reproducible cada ensayo: con la misma
                    # cantidad de partículas, los ensayos solo difieren en c1 y c2.
                    torch.manual_seed(pso_seed + point_index)
                    _, value_history, _ = run_pso_fn(
                        T=iterations,
                        c1=c1,
                        c2=c2,
                        func=function,
                        initial_point=initial_point,
                        n_particles=n_particles,
                    )
            except ValueError as error:
                raise optuna.TrialPruned(str(error)) from error

            final_value = float(value_history[-1].detach().cpu())
            if not math.isfinite(final_value):
                raise optuna.TrialPruned("La corrida produjo un valor no finito.")

            final_values.append(final_value)
            results.append(
                evaluate_convergence(
                    value_history,
                    minimum_value,
                    tolerance,
                )
            )

        converged_runs = [result for result in results if result["converged"]]
        reported_values = [
            result["reported_value"]
            for result in results
            if math.isfinite(result["reported_value"])
        ]
        mean_final_value = float(np.mean(final_values))

        trial.set_user_attr("converged_runs", len(converged_runs))
        trial.set_user_attr("diverged_runs", 0)
        trial.set_user_attr(
            "mean_convergence_iterations",
            (float(np.mean([r["iterations"] for r in converged_runs])) if converged_runs else None),
        )
        trial.set_user_attr(
            "mean_reported_value",
            (float(np.mean(reported_values)) if reported_values else None),
        )
        trial.set_user_attr("mean_final_value", mean_final_value)
        return mean_final_value

    return objective


def _configure_log_axis(axis, distribution):
    """!
    @brief Configura marcas decimales legibles para un parámetro logarítmico.
    @param axis Eje de Matplotlib que se desea configurar.
    @param distribution Distribución FloatDistribution usada por Optuna.
    @return None.
    """
    lower_bound = float(distribution.low)
    upper_bound = float(distribution.high)
    locator = LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12)
    ticks = [
        tick
        for tick in locator.tick_values(lower_bound, upper_bound)
        if lower_bound <= tick <= upper_bound
    ]
    ticks.extend((lower_bound, upper_bound))
    ticks = sorted(set(ticks))

    lower_log = math.log10(lower_bound)
    upper_log = math.log10(upper_bound)
    padding = max(0.04 * (upper_log - lower_log), 0.02)

    axis.set_xscale("log")
    axis.set_xlim(
        10 ** (lower_log - padding),
        10 ** (upper_log + padding),
    )
    axis.xaxis.set_major_locator(FixedLocator(ticks))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.4g}"))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.tick_params(axis="x", labelrotation=30)


def plot_optuna_study(study, function_name, algorithm, parameters):
    """!
    @brief Grafica el progreso del estudio y el efecto de sus parámetros.
    @param study Estudio de Optuna ya ejecutado.
    @param function_name Etiqueta de la función objetivo.
    @param algorithm Nombre del algoritmo calibrado.
    @param parameters Lista de hiperparámetros ajustados.
    @return None.
    """
    completed_trials = [
        trial
        for trial in study.trials
        if (
            trial.state == optuna.trial.TrialState.COMPLETE
            and trial.value is not None
            and np.isfinite(trial.value)
        )
    ]
    if not completed_trials:
        raise ValueError("El estudio no contiene ensayos completos para graficar.")

    numeric_values = [trial.value for trial in completed_trials]
    absolute_values = np.abs(np.asarray(numeric_values))
    positive_absolute_values = absolute_values[absolute_values > 0]
    wide_range = (
        positive_absolute_values.size > 0
        and positive_absolute_values.max() / positive_absolute_values.min() > 100
    )
    y_scale = (
        "log" if wide_range and all(scalar_value > 0 for scalar_value in numeric_values)
        else "symlog" if wide_range else "linear"
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=optuna.exceptions.ExperimentalWarning,
        )
        axis = optuna.visualization.matplotlib.plot_optimization_history(
            study,
            target_name="Promedio de f al final",
        )

    axis.set_title(f"Historial de optimización: {algorithm} sobre {function_name}")
    axis.figure.set_size_inches(8, 4.5)
    if y_scale == "log":
        axis.set_yscale("log")
    elif y_scale == "symlog":
        axis.set_yscale("symlog", linthresh=max(float(positive_absolute_values.min()), 1e-8))
    _show_figure(axis.figure)

    width = 7 if len(parameters) == 1 else 6 * len(parameters)
    figure, axes = plt.subplots(
        1,
        len(parameters),
        figsize=(width, 4.8),
        squeeze=False,
    )
    axes = axes[0]
    best_trial = study.best_trial

    for current_axis, parameter in zip(axes, parameters):
        parameter_values = [trial.params[parameter] for trial in completed_trials]
        current_axis.scatter(
            parameter_values,
            numeric_values,
            color="tab:blue",
            alpha=0.75,
            label="Ensayos",
        )
        current_axis.scatter(
            best_trial.params[parameter],
            best_trial.value,
            marker="*",
            s=160,
            color="crimson",
            edgecolor="black",
            label="Mejor ensayo",
            zorder=3,
        )
        current_axis.set_xlabel(
            "Número de partículas" if parameter == "n_particles" else parameter
        )
        current_axis.grid(alpha=0.3)

        distribution = completed_trials[0].distributions[parameter]
        if getattr(distribution, "log", False):
            _configure_log_axis(current_axis, distribution)
        if y_scale == "log":
            current_axis.set_yscale("log")
        elif y_scale == "symlog":
            current_axis.set_yscale("symlog", linthresh=max(float(positive_absolute_values.min()), 1e-8))

    axes[0].set_ylabel("Promedio de f al final")
    axes[0].legend()
    figure.suptitle(f"Efecto de hiperparámetros: {algorithm} sobre {function_name}")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    _show_figure(figure)


def plot_best_run(
    algorithm,
    function_name,
    best_runs,
    evaluation_histories,
    evaluation_functions,
    x_grid,
    y_grid,
    evaluation_swarms=None,
    tolerance=1e-3,
):
    """!
    @brief Grafica la trayectoria y curva de la corrida más rápida.
    @param algorithm Nombre del algoritmo: GD, RMSProp o PSO.
    @param function_name Identificador de la función: f0, f1 o f2.
    @param best_runs Selecciones calculadas para cada combinación.
    @param evaluation_histories Historiales de puntos y valores por corrida.
    @param evaluation_functions Metadatos de las funciones objetivo.
    @param x_grid Valores del eje x usados para construir la malla.
    @param y_grid Valores del eje y usados para construir la malla.
    @param evaluation_swarms Posiciones de todas las partículas en cada corrida de PSO.
    @param tolerance Brecha usada para dibujar el valor de convergencia de PSO.
    @return None.
    """
    global _figures_prefix
    _figures_prefix = "evaluacion"
    try:
        _plot_best_run_details(
            algorithm,
            function_name,
            best_runs,
            evaluation_histories,
            evaluation_functions,
            x_grid,
            y_grid,
            evaluation_swarms,
            tolerance,
        )
    finally:
        _figures_prefix = ""


def _plot_best_run_details(
    algorithm,
    function_name,
    best_runs,
    evaluation_histories,
    evaluation_functions,
    x_grid,
    y_grid,
    evaluation_swarms,
    tolerance,
):
    """!
    @brief Implementa plot_best_run; las figuras se exportan con prefijo de evaluación.
    @param algorithm Nombre del algoritmo: GD, RMSProp o PSO.
    @param function_name Identificador de la función: f0, f1 o f2.
    @param best_runs Selecciones calculadas para cada combinación.
    @param evaluation_histories Historiales de puntos y valores por corrida.
    @param evaluation_functions Metadatos de las funciones objetivo.
    @param x_grid Valores del eje x usados para construir la malla.
    @param y_grid Valores del eje y usados para construir la malla.
    @param evaluation_swarms Posiciones de las partículas en cada corrida de PSO.
    @param tolerance Brecha usada para dibujar la convergencia de PSO.
    @return None.
    """
    selection = best_runs[(algorithm, function_name)]
    run_number = int(selection["corrida"])
    key = (algorithm, function_name, run_number)
    point_history, value_history = evaluation_histories[key]
    function_data = evaluation_functions[function_name]
    swarm_history_record = None if evaluation_swarms is None else evaluation_swarms.get(key)

    if selection["estado"] == "Convergió":
        final_index = int(selection["índice de reporte"])
        description = f"convergió en {int(selection['iteraciones hasta converger'])} " "iteraciones"
    else:
        final_index = len(point_history) - 1
        description = "no hubo convergencia; se muestra el menor valor final disponible"

    displayed_points = point_history[: final_index + 1]
    displayed_values = value_history[: final_index + 1]
    print(
        f"{algorithm} sobre {function_name}, corrida {run_number}: "
        f"{description}; f={selection['valor reportado']:.6f}."
    )

    trajectory_title = f"Mejor trayectoria de {algorithm} sobre {function_name}"
    if swarm_history_record is not None:
        # Igual que en GD y RMSProp se grafica una sola trayectoria: la de la partícula
        # que llegó al valor más cercano al mínimo, desde su inicio hasta ese punto.
        displayed_swarm_history = swarm_history_record[: final_index + 1]
        particle_index, best_iteration, best_value = best_pso_particle(
            function_data["function"],
            displayed_swarm_history,
        )
        displayed_points = displayed_swarm_history[: best_iteration + 1, particle_index]
        trajectory_title += f" (partícula {particle_index + 1})"
        print(
            f"Mejor partícula: {particle_index + 1} de {displayed_swarm_history.shape[1]}; "
            f"llegó a f={best_value:.6f} en la iteración {best_iteration}."
        )

    plot_func_hist(
        x_grid,
        y_grid,
        function_data["function"],
        displayed_points,
        title=trajectory_title,
        minima=function_data["minima"],
        local_minima=function_data["local_minima"],
    )
    plot_learning_curve(
        displayed_values,
        title=(f"Curva de aprendizaje de {algorithm} sobre {function_name}"),
    )
    if swarm_history_record is not None:
        pso_hyperparameters = {
            "c1": selection["c1"],
            "c2": selection["c2"],
            "n_particles": int(selection["n_particles"]),
        }
        plot_pso_history_groups(
            function_data["function"],
            displayed_swarm_history,
            title=(f"Evolución de f en cada partícula de {algorithm} sobre {function_name}"),
            minima=function_data["minima"],
            tolerance=tolerance,
            hyperparameters=pso_hyperparameters,
        )
        plot_best_pso_particle(
            function_data["function"],
            displayed_swarm_history,
            title=(f"Partícula más cercana al mínimo de {algorithm} sobre {function_name}"),
            minima=function_data["minima"],
            tolerance=tolerance,
            hyperparameters=pso_hyperparameters,
        )
