import torch
import matplotlib.pyplot as plt


def _points_as_tensor(points):
    """Convierte uno o varios puntos bidimensionales en un tensor de forma (n, 2)."""
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

    axis.set_title(title)
    axis.set_xlabel("Iteración t")
    axis.set_ylabel(r"$f(x_t)$")
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
    @brief Grafica la trayectoria del algoritmo sobre las curvas de nivel.
    @param x_grid Valores del eje x usados para construir la malla.
    @param y_grid Valores del eje y usados para construir la malla.
    @param func Función que recibe un tensor con las coordenadas x e y.
    @param visited_points Puntos recorridos por el algoritmo, con forma (n, 2).
    @param title Título de la gráfica.
    @param minima Uno o varios mínimos globales conocidos.
    @param local_minima Mínimos locales que se muestran con marcadores pequeños.
    @param view_limits Límites opcionales (mínimo, máximo) para ambos ejes.
    @return La figura y sus ejes para permitir ajustes posteriores.
    """
    x_values = torch.as_tensor(x_grid).detach().cpu()
    y_values = torch.as_tensor(y_grid).detach().cpu()
    path = _points_as_tensor(visited_points)
    if path is None or path.shape[0] == 0:
        raise ValueError("El historial de puntos no puede estar vacío.")

    X1, X2 = torch.meshgrid(x_values, y_values, indexing="ij")
    with torch.no_grad():
        values = func(torch.stack((X1, X2)))

    # El cambio de escala hace visibles los valles sin modificar la trayectoria.
    color_values = torch.log1p((values - values.min()).clamp_min(0))

    figure, axis = plt.subplots(figsize=(8, 6.5))
    contours = axis.contourf(
        X1.numpy(),
        X2.numpy(),
        color_values.detach().cpu().numpy(),
        levels=50,
        cmap="viridis",
    )
    colorbar = figure.colorbar(contours, ax=axis)
    colorbar.set_label(r"$\log(1 + f(x,y) - f_{min})$")

    path_values = path.numpy()
    marker_interval = max(1, len(path_values) // 50)
    axis.plot(
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

    if view_limits is not None:
        axis.set_xlim(view_limits)
        axis.set_ylim(view_limits)

    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    figure.tight_layout()
    plt.show()
    return figure, axis
