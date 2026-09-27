import math
import numbers
import random
import warnings

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


def _graficar_minimos(axis, minima, local_minima):
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


def _titulo_con_hiperparametros(title, hiperparametros):
    """!
    @brief Agrega al título una línea con los hiperparámetros usados.
    @param title Título principal de la gráfica.
    @param hiperparametros Diccionario opcional con nombre y valor de cada hiperparámetro.
    @return Título con el subtítulo de hiperparámetros, si se proporcionaron.
    """
    if not hiperparametros:
        return title

    # Los enteros, como la cantidad de partículas, se muestran sin decimales.
    texto_hiperparametros = ", ".join(
        f"{nombre}={valor}" if isinstance(valor, numbers.Integral) else f"{nombre}={valor:.5f}"
        for nombre, valor in hiperparametros.items()
    )
    return f"{title}\n{texto_hiperparametros}"


def _estilo_particula(indice_particula):
    """!
    @brief Devuelve el color, el trazo y el marcador asignados a una partícula.
    @param indice_particula Índice de la partícula dentro del enjambre.
    @return Tupla con color, estilo de línea y marcador.
    """
    # Después de diez partículas los colores se repiten y cambian el trazo y el marcador;
    # los cinco estilos distinguen hasta 50 partículas. La paleta fija evita que el estilo
    # activo de Matplotlib (Optuna aplica ggplot) repita colores antes de la décima.
    estilos = [
        ("--", "o"),
        ("-.", "s"),
        (":", "^"),
        ("-", "D"),
        ((0, (5, 1, 1, 1, 1, 1)), "v"),
    ]
    color = matplotlib.colormaps["tab10"].colors[indice_particula % 10]
    estilo_linea, marcador = estilos[(indice_particula // 10) % len(estilos)]
    return color, estilo_linea, marcador


def _leyenda_exterior(figure, axis, filas_maximas=25):
    """!
    @brief Coloca la leyenda a la derecha de la figura y la reparte en columnas si es larga.
    @param figure Figura de Matplotlib que recibe la leyenda.
    @param axis Eje cuyos elementos con etiqueta forman la leyenda.
    @param filas_maximas Filas por columna antes de agregar otra columna.
    @return None.
    """
    cantidad_entradas = len(axis.get_legend_handles_labels()[1])
    columnas = max(1, math.ceil(cantidad_entradas / filas_maximas))
    figure.legend(
        loc="outside right upper",
        ncols=columnas,
        fontsize="small" if columnas > 1 else "medium",
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
    plt.show()


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
    magnitudes = np.abs(plot_values[plot_values != 0])
    if magnitudes.size and magnitudes.max() / magnitudes.min() >= 100:
        if np.all(plot_values > 0):
            axis.set_yscale("log")
        else:
            axis.set_yscale("symlog", linthresh=max(float(magnitudes.min()), 1e-8))
        axis.set_ylabel(r"$f(x_t)$ (escala logarítmica)")
    else:
        axis.set_ylabel(r"$f(x_t)$")

    axis.set_title(title)
    axis.set_xlabel("Iteración t")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    plt.show()
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

    limites_originales_x = (float(x_values.min()), float(x_values.max()))
    limites_originales_y = (float(y_values.min()), float(y_values.max()))
    x_values_originales = x_values
    y_values_originales = y_values
    puntos_visibles = [path.numpy()]
    for puntos_conocidos in (minima, local_minima):
        tensor_puntos = _points_as_tensor(puntos_conocidos)
        if tensor_puntos is not None:
            puntos_visibles.append(tensor_puntos.numpy())
    puntos_visibles = np.concatenate(puntos_visibles)

    limite_x_inferior = min(limites_originales_x[0], float(puntos_visibles[:, 0].min()))
    limite_x_superior = max(limites_originales_x[1], float(puntos_visibles[:, 0].max()))
    limite_y_inferior = min(limites_originales_y[0], float(puntos_visibles[:, 1].min()))
    limite_y_superior = max(limites_originales_y[1], float(puntos_visibles[:, 1].max()))
    malla_ampliada = view_limits is None and (
        limite_x_inferior < limites_originales_x[0]
        or limite_x_superior > limites_originales_x[1]
        or limite_y_inferior < limites_originales_y[0]
        or limite_y_superior > limites_originales_y[1]
    )
    if malla_ampliada:
        # El margen depende del recorrido; se conserva la cantidad original de muestras.
        margen_x = 0.05 * (limite_x_superior - limite_x_inferior)
        margen_y = 0.05 * (limite_y_superior - limite_y_inferior)
        x_values = torch.linspace(
            limite_x_inferior - margen_x,
            limite_x_superior + margen_x,
            x_values.numel(),
            dtype=x_values.dtype,
        )
        y_values = torch.linspace(
            limite_y_inferior - margen_y,
            limite_y_superior + margen_y,
            y_values.numel(),
            dtype=y_values.dtype,
        )

    X1, X2 = torch.meshgrid(x_values, y_values, indexing="ij")
    with torch.no_grad():
        values = func(torch.stack((X1, X2)))
        if malla_ampliada:
            X1_original, X2_original = torch.meshgrid(
                x_values_originales, y_values_originales, indexing="ij"
            )
            valores_originales = func(torch.stack((X1_original, X2_original)))

    # El cambio de escala hace visibles los valles sin modificar la trayectoria.
    valor_minimo = values.min()
    niveles = 50
    extension = "neither"
    if malla_ampliada:
        valor_minimo = torch.minimum(valor_minimo, valores_originales.min())
        # La escala conserva el contraste del dominio de referencia aunque
        # aparezcan valores mayores en la zona nueva.
        maximo_referencia = torch.log1p(
            (valores_originales - valor_minimo).clamp_min(0)
        ).max().item()
        if maximo_referencia > 0:
            niveles = np.linspace(0, maximo_referencia, 51)
            extension = "max"
    color_values = torch.log1p((values - valor_minimo).clamp_min(0))

    figure, axis = plt.subplots(figsize=(8, 6.5))
    contours = axis.contourf(
        X1.numpy(),
        X2.numpy(),
        color_values.detach().cpu().numpy(),
        levels=niveles,
        cmap="viridis",
        extend=extension,
    )
    colorbar = figure.colorbar(contours, ax=axis)
    colorbar.set_label(r"$\log(1 + f(x,y) - f_{min})$")

    if malla_ampliada:
        axis.add_patch(
            Rectangle(
                (limites_originales_x[0], limites_originales_y[0]),
                limites_originales_x[1] - limites_originales_x[0],
                limites_originales_y[1] - limites_originales_y[0],
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

    _graficar_minimos(axis, minima, local_minima)

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
    plt.show()
    return figure, axis


def plot_historial_pso(
    func,
    enjambre_hist,
    title,
    minima,
    tolerancia=1e-3,
    hiperparametros=None,
    particulas=None,
    resaltar_minimo=False,
):
    """!
    @brief Grafica en escala lineal las partículas de PSO seleccionadas.
    @param func Función que recibe un tensor con las coordenadas x e y.
    @param enjambre_hist Posiciones de todas las partículas, con forma (T+1, n, 2).
    @param title Título de la gráfica.
    @param minima Uno o varios mínimos globales calculados para la función.
    @param tolerancia Brecha máxima respecto al mínimo para considerar convergencia.
    @param hiperparametros Diccionario opcional con los hiperparámetros del subtítulo.
    @param particulas Índices opcionales de las partículas graficadas; por omisión, todas.
    @param resaltar_minimo Si es True, marca el menor valor de las partículas graficadas.
    @return La figura y sus ejes para permitir ajustes posteriores.
    """
    enjambre = torch.as_tensor(enjambre_hist).detach().cpu()
    if enjambre.ndim != 3 or enjambre.shape[2] != 2:
        raise ValueError("El historial del enjambre debe tener forma (T+1, n, 2).")
    minima_tensor = _points_as_tensor(minima)
    if minima_tensor is None:
        raise ValueError("Se necesita al menos un mínimo global de la función.")

    # El mínimo global se evalúa en los mínimos calculados, no se asume igual a 0.
    with torch.no_grad():
        valores = func(enjambre.permute(2, 0, 1))
        valor_minimo = func(minima_tensor.T).min().item()
    if not torch.isfinite(valores).all():
        raise ValueError("El historial contiene NaN o infinito; PSO pudo divergir.")

    valores = valores.numpy()
    posiciones = enjambre.numpy()
    n_iteraciones, n_particulas = valores.shape
    indices = list(range(n_particulas)) if particulas is None else list(particulas)
    iteraciones = np.arange(n_iteraciones)
    valor_convergencia = valor_minimo + tolerancia

    figure, axis = plt.subplots(
        figsize=(13, 6) if len(indices) > 1 else (12, 6),
        layout="constrained",
    )
    axis.axhline(
        valor_convergencia,
        color="darkred",
        linestyle=":",
        linewidth=1.5,
        label=f"Convergencia = {valor_convergencia:g}",
        zorder=4,
    )

    # Cada partícula conserva su color y trazo aunque solo se grafique una parte del enjambre.
    intervalo_marcadores = max(1, n_iteraciones // 12)
    for indice_particula in indices:
        color, estilo_linea, marcador = _estilo_particula(indice_particula)
        x_0, y_0 = posiciones[0, indice_particula]
        axis.plot(
            iteraciones,
            valores[:, indice_particula],
            color=color,
            linestyle=estilo_linea,
            marker=marcador,
            markevery=intervalo_marcadores,
            markersize=3,
            linewidth=1.3,
            alpha=0.8,
            label=f"Partícula {indice_particula + 1}: ({x_0:.2f}, {y_0:.2f})",
        )

    if resaltar_minimo:
        valores_graficados = valores[:, indices]
        iteracion, columna = np.unravel_index(
            np.argmin(valores_graficados),
            valores_graficados.shape,
        )
        valor_resaltado = valores_graficados[iteracion, columna]
        x_min, y_min = posiciones[iteracion, indices[columna]]
        axis.scatter(
            iteracion,
            valor_resaltado,
            marker="*",
            s=320,
            color="gold",
            edgecolor="black",
            label=(
                f"Más cercano al mínimo: f = {valor_resaltado:.4g} en t = {iteracion}\n"
                f"(x, y) = ({x_min:.4f}, {y_min:.4f})"
            ),
            zorder=5,
        )

    axis.set_title(_titulo_con_hiperparametros(title, hiperparametros))
    axis.set_xlabel("Iteración")
    axis.set_ylabel(r"$f(x, y)$")
    axis.grid(alpha=0.3)
    _leyenda_exterior(figure, axis)
    plt.show()
    return figure, axis


def plot_historial_pso_en_grupos(
    func,
    enjambre_hist,
    title,
    minima,
    tolerancia=1e-3,
    hiperparametros=None,
):
    """!
    @brief Grafica el enjambre de PSO en grupos consecutivos de hasta diez partículas.
    @param func Función que recibe un tensor con las coordenadas x e y.
    @param enjambre_hist Posiciones de todas las partículas, con forma (T+1, n, 2).
    @param title Título común de las gráficas.
    @param minima Uno o varios mínimos globales calculados para la función.
    @param tolerancia Brecha máxima respecto al mínimo para considerar convergencia.
    @param hiperparametros Diccionario opcional con los hiperparámetros del subtítulo.
    @return Lista de pares (figura, ejes), uno por cada grupo de partículas.
    """
    enjambre = torch.as_tensor(enjambre_hist)
    if enjambre.ndim != 3 or enjambre.shape[2] != 2 or enjambre.shape[1] == 0:
        raise ValueError("El historial del enjambre debe tener forma (T+1, n, 2) con n > 0.")

    graficas = []
    for inicio in range(0, enjambre.shape[1], 10):
        fin = min(inicio + 10, enjambre.shape[1])
        graficas.append(
            plot_historial_pso(
                func,
                enjambre,
                title=f"{title} (partículas {inicio + 1}–{fin})",
                minima=minima,
                tolerancia=tolerancia,
                hiperparametros=hiperparametros,
                particulas=range(inicio, fin),
            )
        )
    return graficas


def mejor_particula_pso(func, enjambre_hist):
    """!
    @brief Encuentra la partícula que llegó al valor más cercano al mínimo global.
    @param func Función que recibe un tensor con las coordenadas x e y.
    @param enjambre_hist Posiciones de todas las partículas, con forma (T+1, n, 2).
    @return Tupla con el índice de la partícula, la iteración y el menor valor alcanzado.
    """
    enjambre = torch.as_tensor(enjambre_hist).detach().cpu()
    if enjambre.ndim != 3 or enjambre.shape[2] != 2:
        raise ValueError("El historial del enjambre debe tener forma (T+1, n, 2).")

    with torch.no_grad():
        valores = func(enjambre.permute(2, 0, 1))
    if not torch.isfinite(valores).all():
        raise ValueError("El historial contiene NaN o infinito; PSO pudo divergir.")

    # Como f nunca es menor que su mínimo global, el menor valor es el más cercano a él.
    iteracion, indice_particula = divmod(int(torch.argmin(valores)), valores.shape[1])
    return indice_particula, iteracion, float(valores[iteracion, indice_particula])


def plot_mejor_particula_pso(
    func,
    enjambre_hist,
    title,
    minima,
    tolerancia=1e-3,
    hiperparametros=None,
):
    """!
    @brief Grafica la evolución de f en la partícula que llegó más cerca del mínimo global.
    @param func Función que recibe un tensor con las coordenadas x e y.
    @param enjambre_hist Posiciones de todas las partículas, con forma (T+1, n, 2).
    @param title Título de la gráfica.
    @param minima Uno o varios mínimos globales calculados para la función.
    @param tolerancia Brecha máxima respecto al mínimo para considerar convergencia.
    @param hiperparametros Diccionario opcional con los hiperparámetros del subtítulo.
    @return La figura y sus ejes para permitir ajustes posteriores.
    """
    indice_particula, _, _ = mejor_particula_pso(func, enjambre_hist)
    return plot_historial_pso(
        func,
        enjambre_hist,
        title=f"{title} (partícula {indice_particula + 1})",
        minima=minima,
        tolerancia=tolerancia,
        hiperparametros=hiperparametros,
        particulas=[indice_particula],
        resaltar_minimo=True,
    )


def plot_curvas_aprendizaje(
    historiales_valores,
    etiquetas,
    title,
    valor_minimo,
    tolerancia=1e-3,
    hiperparametros=None,
):
    """!
    @brief Grafica varias curvas de aprendizaje y su promedio en escala lineal.
    @param historiales_valores Historiales f(x_t) de igual longitud, uno por corrida.
    @param etiquetas Texto de la leyenda para cada historial.
    @param title Título de la gráfica.
    @param valor_minimo Valor mínimo global conocido de la función.
    @param tolerancia Brecha máxima respecto al mínimo para considerar convergencia.
    @param hiperparametros Diccionario opcional con los hiperparámetros del subtítulo.
    @return La figura y sus ejes para permitir ajustes posteriores.
    """
    curvas = torch.stack(
        [torch.as_tensor(historial).detach().cpu().flatten() for historial in historiales_valores]
    )
    if curvas.shape[0] != len(etiquetas):
        raise ValueError("Se necesita una etiqueta por cada curva de aprendizaje.")
    if not torch.isfinite(curvas).all():
        raise ValueError("Los historiales contienen NaN o infinito.")

    curvas = curvas.numpy()
    iteraciones = np.arange(curvas.shape[1])
    valor_convergencia = valor_minimo + tolerancia

    figure, axis = plt.subplots(figsize=(13, 6), layout="constrained")
    axis.axhline(
        valor_convergencia,
        color="darkred",
        linestyle=":",
        linewidth=1.5,
        label=f"Convergencia = {valor_convergencia:g}",
        zorder=4,
    )
    intervalo_marcadores = max(1, len(iteraciones) // 12)
    for indice, (curva, etiqueta) in enumerate(zip(curvas, etiquetas)):
        axis.plot(
            iteraciones,
            curva,
            color=f"C{indice % 10}",
            linestyle="--",
            marker="o",
            markevery=intervalo_marcadores,
            markersize=3,
            linewidth=1.2,
            alpha=0.75,
            label=etiqueta,
        )
    axis.plot(
        iteraciones,
        curvas.mean(axis=0),
        color="black",
        linewidth=2.5,
        label="Promedio",
    )

    axis.set_title(_titulo_con_hiperparametros(title, hiperparametros))
    axis.set_xlabel("Iteración")
    axis.set_ylabel(r"$f(x_t)$")
    axis.grid(alpha=0.3)
    figure.legend(loc="outside right upper")
    plt.show()
    return figure, axis


def mostrar_tabla(tabla, titulo, formatos=None, na_rep="—"):
    """!
    @brief Muestra un DataFrame con un título visible y formato consistente.
    @param tabla DataFrame de pandas que se desea mostrar.
    @param titulo Título que aparecerá como encabezado de la tabla.
    @param formatos Mapeo opcional de columnas a formatos de pandas Styler.
    @param na_rep Texto usado para representar valores ausentes.
    @return None.
    """
    if not isinstance(tabla, pd.DataFrame):
        raise TypeError("La tabla debe ser un DataFrame de pandas.")

    estilo = tabla.style.set_caption(titulo).set_table_styles(
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
    estilo = estilo.format(formatos, na_rep=na_rep)
    display(estilo)


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
    plt.show()


def coordenadas_minimo_malla(x_grid, y_grid, values):
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


def gradiente_y_hessiana(funcion, punto):
    """!
    @brief Calcula valor, gradiente y Hessiana de una función en un punto.
    @param funcion Función de dos variables implementada con PyTorch.
    @param punto Tensor de dos componentes con requires_grad=True.
    @return Tupla con valor, gradiente y matriz Hessiana.
    """
    valor = funcion(punto)
    gradiente = torch.autograd.grad(valor, punto, create_graph=True)[0]
    filas_hessiana = [
        torch.autograd.grad(componente, punto, retain_graph=True)[0] for componente in gradiente
    ]
    return valor, gradiente, torch.stack(filas_hessiana)


def encontrar_punto_estacionario(
    funcion,
    inicio,
    max_iteraciones=20,
    tolerancia=1e-8,
):
    """!
    @brief Refina un punto inicial hasta que su gradiente sea cercano a cero.
    @param funcion Función de dos variables implementada con PyTorch.
    @param inicio Tupla con las coordenadas iniciales.
    @param max_iteraciones Máximo de pasos de Newton.
    @param tolerancia Norma máxima del gradiente para detener el método.
    @return Tensor con el punto estacionario aproximado.
    """
    punto = torch.tensor(inicio, dtype=torch.float64, requires_grad=True)

    for _ in range(max_iteraciones):
        _, gradiente, hessiana = gradiente_y_hessiana(funcion, punto)
        if torch.linalg.vector_norm(gradiente) < tolerancia:
            break

        paso = torch.linalg.solve(hessiana, gradiente)
        with torch.no_grad():
            punto -= paso

    return punto.detach()


def clasificar_punto(funcion, punto, tolerancia=1e-6):
    """!
    @brief Clasifica un punto mediante el gradiente y la Hessiana 2x2.
    @param funcion Función de dos variables implementada con PyTorch.
    @param punto Tensor con las coordenadas del candidato.
    @param tolerancia Tolerancia usada en las comparaciones numéricas.
    @return Diccionario con coordenadas, valor y comprobaciones.
    """
    punto_autograd = punto.clone().detach().requires_grad_(True)
    valor, gradiente, hessiana = gradiente_y_hessiana(funcion, punto_autograd)
    norma_gradiente = torch.linalg.vector_norm(gradiente).item()
    determinante = torch.det(hessiana).item()
    f_xx = hessiana[0, 0].item()

    if norma_gradiente > tolerancia:
        tipo = "no convergió"
    elif determinante < -tolerancia:
        tipo = "punto silla"
    elif determinante > tolerancia and f_xx > 0:
        tipo = "mínimo local"
    elif determinante > tolerancia and f_xx < 0:
        tipo = "máximo local"
    else:
        tipo = "prueba inconclusa"

    return {
        "punto": punto.detach(),
        "tipo": tipo,
        "x": punto[0].item(),
        "y": punto[1].item(),
        "valor": valor.item(),
        "norma_gradiente": norma_gradiente,
        "det_hessiana": determinante,
    }


def resultados_a_tabla(resultados):
    """!
    @brief Convierte resultados de puntos críticos en una tabla compacta.
    @param resultados Lista de diccionarios creados por clasificar_punto.
    @return DataFrame con los valores necesarios para verificar cada punto.
    """
    columnas = [
        "tipo",
        "x",
        "y",
        "valor",
        "norma_gradiente",
        "det_hessiana",
    ]
    filas = [{columna: resultado[columna] for columna in columnas} for resultado in resultados]
    tabla = pd.DataFrame(filas)
    tabla["norma_gradiente"] = tabla["norma_gradiente"].map(lambda value: f"{value:.2e}")
    return tabla.round(
        {
            "x": 6,
            "y": 6,
            "valor": 6,
            "det_hessiana": 6,
        }
    )


def encontrar_minimos_ackley_en_dominio(
    funcion,
    limite_inferior,
    limite_superior,
):
    """!
    @brief Busca mínimos locales de Ackley desde una cuadrícula de enteros.
    @param funcion Implementación de la función de Ackley.
    @param limite_inferior Extremo inferior del dominio en ambos ejes.
    @param limite_superior Extremo superior del dominio en ambos ejes.
    @return Lista sin duplicados de mínimos locales dentro del dominio.
    """
    minimos = []
    for x0 in range(math.ceil(limite_inferior), math.floor(limite_superior) + 1):
        for y0 in range(math.ceil(limite_inferior), math.floor(limite_superior) + 1):
            if x0 == 0 and y0 == 0:
                continue

            punto = encontrar_punto_estacionario(
                funcion,
                inicio=(float(x0), float(y0)),
            )
            resultado = clasificar_punto(funcion, punto)
            dentro_del_dominio = torch.all(
                (punto >= limite_inferior) & (punto <= limite_superior)
            ).item()
            repetido = any(
                torch.linalg.vector_norm(punto - anterior).item() < 1e-5 for anterior in minimos
            )

            if resultado["tipo"] == "mínimo local" and dentro_del_dominio and not repetido:
                minimos.append(punto)

    return minimos


def valor_y_gradiente(funcion, punto):
    """!
    @brief Calcula el valor de una función y su gradiente en un punto.
    @param funcion Función objetivo implementada con PyTorch.
    @param punto Tensor con respecto al cual se calcula el gradiente.
    @return Tupla con el valor de la función y su gradiente.
    """
    valor = funcion(punto)
    gradiente = torch.autograd.grad(valor, punto)[0]
    return valor, gradiente


def generar_punto_inicial(limite_inferior, limite_superior, generador=None):
    """!
    @brief Genera un punto uniforme al azar dentro de un dominio cuadrado.
    @param limite_inferior Extremo inferior del dominio.
    @param limite_superior Extremo superior del dominio.
    @param generador Generador opcional compatible con random.Random.
    @return Tupla con dos coordenadas aleatorias.
    """
    fuente = generador if generador is not None else random
    return (
        fuente.uniform(limite_inferior, limite_superior),
        fuente.uniform(limite_inferior, limite_superior),
    )


def evaluar_convergencia(valores_historial, valor_minimo, tolerancia):
    """!
    @brief Localiza la primera iteración que satisface la tolerancia objetivo.
    @param valores_historial Valores de la función para cada punto visitado.
    @param valor_minimo Valor mínimo global conocido de la función.
    @param tolerancia Brecha máxima permitida respecto al mínimo.
    @return Diccionario con estado, iteración, índice, valor y brecha reportados.
    """
    for indice, valor in enumerate(valores_historial):
        valor_float = float(valor.detach().cpu())
        if not math.isfinite(valor_float):
            return {
                "convergio": False,
                "divergio": True,
                "iteraciones": None,
                "indice_reporte": indice,
                "valor_reportado": math.nan,
                "brecha": math.inf,
            }

        brecha = abs(valor_float - valor_minimo)
        if brecha <= tolerancia:
            return {
                "convergio": True,
                "divergio": False,
                "iteraciones": indice,
                "indice_reporte": indice,
                "valor_reportado": valor_float,
                "brecha": brecha,
            }

    valor_final = float(valores_historial[-1].detach().cpu())
    return {
        "convergio": False,
        "divergio": not math.isfinite(valor_final),
        "iteraciones": None,
        "indice_reporte": len(valores_historial) - 1,
        "valor_reportado": valor_final,
        "brecha": abs(valor_final - valor_minimo),
    }


def crear_objetivo_optuna(
    algoritmo,
    configuracion,
    puntos_iniciales,
    iteraciones,
    tolerancia,
    epsilon_rmsprop,
    ejecutar_gd,
    ejecutar_rmsprop,
    ejecutar_pso=None,
    semilla_pso=0,
):
    """!
    @brief Crea un objetivo Optuna basado en el promedio del valor final.
    @param algoritmo Nombre del algoritmo: GD, RMSProp o PSO.
    @param configuracion Función, mínimo conocido y rangos de búsqueda.
    @param puntos_iniciales Puntos compartidos por todos los ensayos.
    @param iteraciones Cantidad fija de actualizaciones por corrida.
    @param tolerancia Tolerancia usada solo para métricas diagnósticas.
    @param epsilon_rmsprop Constante de estabilidad fija de RMSProp.
    @param ejecutar_gd Función que ejecuta descenso del gradiente.
    @param ejecutar_rmsprop Función que ejecuta RMSProp.
    @param ejecutar_pso Función que ejecuta el enjambre de partículas.
    @param semilla_pso Semilla base de PSO; cada punto usa semilla_pso + su índice.
    @return Función objetivo compatible con Optuna.
    """
    if algoritmo == "PSO" and ejecutar_pso is None:
        raise ValueError("Se necesita ejecutar_pso para calibrar PSO.")

    funcion = configuracion["funcion"]
    valor_minimo = configuracion["valor_minimo"]

    def objetivo(trial):
        """!
        @brief Evalúa un ensayo con el presupuesto fijo del estudio.
        @param trial Ensayo de Optuna que contiene los hiperparámetros sugeridos.
        @return Promedio del valor final de la función en los puntos iniciales.
        """
        if algoritmo == "GD":
            alpha = trial.suggest_float(
                "alpha",
                configuracion["gd_alpha_min"],
                configuracion["gd_alpha_max"],
                log=True,
            )
            gamma = None
        elif algoritmo == "RMSProp":
            alpha = trial.suggest_float(
                "alpha",
                configuracion["rms_alpha_min"],
                configuracion["rms_alpha_max"],
                log=True,
            )
            gamma = trial.suggest_float(
                "gamma",
                configuracion["gamma_min"],
                configuracion["gamma_max"],
            )
        elif algoritmo == "PSO":
            c1 = trial.suggest_float(
                "c1",
                configuracion["pso_c1_min"],
                configuracion["pso_c1_max"],
            )
            c2 = trial.suggest_float(
                "c2",
                configuracion["pso_c2_min"],
                configuracion["pso_c2_max"],
            )
            n_particulas = trial.suggest_int(
                "n_particulas",
                configuracion["pso_particulas_min"],
                configuracion["pso_particulas_max"],
            )
        else:
            raise ValueError(f"Algoritmo no soportado: {algoritmo}")

        resultados = []
        valores_finales = []

        for indice_punto, punto_inicial in enumerate(puntos_iniciales):
            try:
                if algoritmo == "GD":
                    _, valores_historial = ejecutar_gd(
                        alpha=alpha,
                        t=iteraciones,
                        func=funcion,
                        punto_inicial=punto_inicial,
                    )
                elif algoritmo == "RMSProp":
                    _, valores_historial = ejecutar_rmsprop(
                        alpha=alpha,
                        gamma=gamma,
                        epsilon=epsilon_rmsprop,
                        t=iteraciones,
                        func=funcion,
                        punto_inicial=punto_inicial,
                    )
                else:
                    # La semilla por punto hace reproducible cada ensayo: con la misma
                    # cantidad de partículas, los ensayos solo difieren en c1 y c2.
                    torch.manual_seed(semilla_pso + indice_punto)
                    _, valores_historial, _ = ejecutar_pso(
                        T=iteraciones,
                        c1=c1,
                        c2=c2,
                        func=funcion,
                        punto_inicial=punto_inicial,
                        n_particulas=n_particulas,
                    )
            except ValueError as error:
                raise optuna.TrialPruned(str(error)) from error

            valor_final = float(valores_historial[-1].detach().cpu())
            if not math.isfinite(valor_final):
                raise optuna.TrialPruned("La corrida produjo un valor no finito.")

            valores_finales.append(valor_final)
            resultados.append(
                evaluar_convergencia(
                    valores_historial,
                    valor_minimo,
                    tolerancia,
                )
            )

        convergentes = [resultado for resultado in resultados if resultado["convergio"]]
        valores_reportados = [
            resultado["valor_reportado"]
            for resultado in resultados
            if math.isfinite(resultado["valor_reportado"])
        ]
        promedio_final = float(np.mean(valores_finales))

        trial.set_user_attr("corridas_convergentes", len(convergentes))
        trial.set_user_attr("corridas_divergentes", 0)
        trial.set_user_attr(
            "iteraciones_promedio_convergencia",
            (float(np.mean([r["iteraciones"] for r in convergentes])) if convergentes else None),
        )
        trial.set_user_attr(
            "promedio_valor_reportado",
            (float(np.mean(valores_reportados)) if valores_reportados else None),
        )
        trial.set_user_attr("promedio_valor_final", promedio_final)
        return promedio_final

    return objetivo


def _configurar_eje_logaritmico(axis, distribucion):
    """!
    @brief Configura marcas decimales legibles para un parámetro logarítmico.
    @param axis Eje de Matplotlib que se desea configurar.
    @param distribucion Distribución FloatDistribution usada por Optuna.
    @return None.
    """
    limite_inferior = float(distribucion.low)
    limite_superior = float(distribucion.high)
    localizador = LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12)
    marcas = [
        marca
        for marca in localizador.tick_values(limite_inferior, limite_superior)
        if limite_inferior <= marca <= limite_superior
    ]
    marcas.extend((limite_inferior, limite_superior))
    marcas = sorted(set(marcas))

    log_inferior = math.log10(limite_inferior)
    log_superior = math.log10(limite_superior)
    margen = max(0.04 * (log_superior - log_inferior), 0.02)

    axis.set_xscale("log")
    axis.set_xlim(
        10 ** (log_inferior - margen),
        10 ** (log_superior + margen),
    )
    axis.xaxis.set_major_locator(FixedLocator(marcas))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.4g}"))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.tick_params(axis="x", labelrotation=30)


def plot_estudio_optuna(estudio, nombre_funcion, algoritmo, parametros):
    """!
    @brief Grafica el progreso del estudio y el efecto de sus parámetros.
    @param estudio Estudio de Optuna ya ejecutado.
    @param nombre_funcion Etiqueta de la función objetivo.
    @param algoritmo Nombre del algoritmo calibrado.
    @param parametros Lista de hiperparámetros ajustados.
    @return None.
    """
    ensayos = [
        trial
        for trial in estudio.trials
        if (
            trial.state == optuna.trial.TrialState.COMPLETE
            and trial.value is not None
            and np.isfinite(trial.value)
        )
    ]
    if not ensayos:
        raise ValueError("El estudio no contiene ensayos completos para graficar.")

    valores = [trial.value for trial in ensayos]
    magnitudes = np.abs(np.asarray(valores))
    magnitudes_positivas = magnitudes[magnitudes > 0]
    escala_amplia = (
        magnitudes_positivas.size > 0
        and magnitudes_positivas.max() / magnitudes_positivas.min() > 100
    )
    escala_y = (
        "log" if escala_amplia and all(valor > 0 for valor in valores)
        else "symlog" if escala_amplia else "linear"
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=optuna.exceptions.ExperimentalWarning,
        )
        axis = optuna.visualization.matplotlib.plot_optimization_history(
            estudio,
            target_name="Promedio de f al final",
        )

    axis.set_title(f"Historial de optimización: {algoritmo} sobre {nombre_funcion}")
    axis.figure.set_size_inches(8, 4.5)
    if escala_y == "log":
        axis.set_yscale("log")
    elif escala_y == "symlog":
        axis.set_yscale("symlog", linthresh=max(float(magnitudes_positivas.min()), 1e-8))
    plt.show()

    ancho = 7 if len(parametros) == 1 else 6 * len(parametros)
    figure, axes = plt.subplots(
        1,
        len(parametros),
        figsize=(ancho, 4.8),
        squeeze=False,
    )
    axes = axes[0]
    mejor_ensayo = estudio.best_trial

    for current_axis, parametro in zip(axes, parametros):
        valores_parametro = [trial.params[parametro] for trial in ensayos]
        current_axis.scatter(
            valores_parametro,
            valores,
            color="tab:blue",
            alpha=0.75,
            label="Ensayos",
        )
        current_axis.scatter(
            mejor_ensayo.params[parametro],
            mejor_ensayo.value,
            marker="*",
            s=160,
            color="crimson",
            edgecolor="black",
            label="Mejor ensayo",
            zorder=3,
        )
        current_axis.set_xlabel(parametro)
        current_axis.grid(alpha=0.3)

        distribucion = ensayos[0].distributions[parametro]
        if getattr(distribucion, "log", False):
            _configurar_eje_logaritmico(current_axis, distribucion)
        if escala_y == "log":
            current_axis.set_yscale("log")
        elif escala_y == "symlog":
            current_axis.set_yscale("symlog", linthresh=max(float(magnitudes_positivas.min()), 1e-8))

    axes[0].set_ylabel("Promedio de f al final")
    axes[0].legend()
    figure.suptitle(f"Efecto de hiperparámetros: {algoritmo} sobre {nombre_funcion}")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    plt.show()


def graficar_mejor_corrida(
    algoritmo,
    nombre_funcion,
    mejores_corridas,
    historiales_evaluacion,
    funciones_evaluacion,
    x_grid,
    y_grid,
    enjambres_evaluacion=None,
    tolerancia=1e-3,
):
    """!
    @brief Grafica la trayectoria y curva de la corrida más rápida.
    @param algoritmo Nombre del algoritmo: GD, RMSProp o PSO.
    @param nombre_funcion Identificador de la función: f0, f1 o f2.
    @param mejores_corridas Selecciones calculadas para cada combinación.
    @param historiales_evaluacion Historiales de puntos y valores por corrida.
    @param funciones_evaluacion Metadatos de las funciones objetivo.
    @param x_grid Valores del eje x usados para construir la malla.
    @param y_grid Valores del eje y usados para construir la malla.
    @param enjambres_evaluacion Posiciones de todas las partículas en cada corrida de PSO.
    @param tolerancia Brecha usada para dibujar el valor de convergencia de PSO.
    @return None.
    """
    seleccion = mejores_corridas[(algoritmo, nombre_funcion)]
    corrida = int(seleccion["corrida"])
    clave = (algoritmo, nombre_funcion, corrida)
    puntos_historial, valores_historial = historiales_evaluacion[clave]
    datos_funcion = funciones_evaluacion[nombre_funcion]
    enjambre_historial = None if enjambres_evaluacion is None else enjambres_evaluacion.get(clave)

    if seleccion["estado"] == "Convergió":
        indice_final = int(seleccion["índice de reporte"])
        descripcion = f"convergió en {int(seleccion['iteraciones hasta converger'])} " "iteraciones"
    else:
        indice_final = len(puntos_historial) - 1
        descripcion = "no hubo convergencia; se muestra el menor valor final disponible"

    puntos_mostrados = puntos_historial[: indice_final + 1]
    valores_mostrados = valores_historial[: indice_final + 1]
    print(
        f"{algoritmo} sobre {nombre_funcion}, corrida {corrida}: "
        f"{descripcion}; f={seleccion['valor reportado']:.6f}."
    )

    titulo_trayectoria = f"Mejor trayectoria de {algoritmo} sobre {nombre_funcion}"
    if enjambre_historial is not None:
        # Igual que en GD y RMSProp se grafica una sola trayectoria: la de la partícula
        # que llegó al valor más cercano al mínimo, desde su inicio hasta ese punto.
        enjambre_mostrado = enjambre_historial[: indice_final + 1]
        indice_particula, iteracion_mejor, valor_mejor = mejor_particula_pso(
            datos_funcion["funcion"],
            enjambre_mostrado,
        )
        puntos_mostrados = enjambre_mostrado[: iteracion_mejor + 1, indice_particula]
        titulo_trayectoria += f" (partícula {indice_particula + 1})"
        print(
            f"Mejor partícula: {indice_particula + 1} de {enjambre_mostrado.shape[1]}; "
            f"llegó a f={valor_mejor:.6f} en la iteración {iteracion_mejor}."
        )

    plot_func_hist(
        x_grid,
        y_grid,
        datos_funcion["funcion"],
        puntos_mostrados,
        title=titulo_trayectoria,
        minima=datos_funcion["minimos"],
        local_minima=datos_funcion["minimos_locales"],
    )
    plot_learning_curve(
        valores_mostrados,
        title=(f"Curva de aprendizaje de {algoritmo} sobre {nombre_funcion}"),
    )
    if enjambre_historial is not None:
        hiperparametros_pso = {
            "c1": seleccion["c1"],
            "c2": seleccion["c2"],
            "n_particulas": int(seleccion["n_particulas"]),
        }
        plot_historial_pso_en_grupos(
            datos_funcion["funcion"],
            enjambre_mostrado,
            title=(f"Evolución de f en cada partícula de {algoritmo} sobre {nombre_funcion}"),
            minima=datos_funcion["minimos"],
            tolerancia=tolerancia,
            hiperparametros=hiperparametros_pso,
        )
        plot_mejor_particula_pso(
            datos_funcion["funcion"],
            enjambre_mostrado,
            title=(f"Partícula más cercana al mínimo de {algoritmo} sobre {nombre_funcion}"),
            minima=datos_funcion["minimos"],
            tolerancia=tolerancia,
            hiperparametros=hiperparametros_pso,
        )
