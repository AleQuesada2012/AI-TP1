import torch
import matplotlib.pyplot as plt


def gradiente_f1(punto: torch.Tensor) -> torch.Tensor:
    #Gradiente de f1 evaluado en punto = (x, y).

    x = punto[0]
    y = punto[1]
    #Sacar tanto el x como el y del punto

    r = torch.sqrt(0.5 * (x**2 + y**2))
    exp_a = torch.exp(-0.2 * r)
    exp_b = torch.exp(0.5 * (torch.cos(2 * torch.pi * x) + torch.cos(2 * torch.pi * y)))
    #Calcular ambas exponenciaciones utilizando torch

    derivada_terminosx = (2 * x / r) * exp_a + torch.pi * torch.sin(2 * torch.pi * x) * exp_b
    derivada_terminosy = (2 * y / r) * exp_a + torch.pi * torch.sin(2 * torch.pi * y) * exp_b
    #Realiza el calculo de la derivada

    return torch.tensor([derivada_terminosx, derivada_terminosy])


def gradiente_f2(punto: torch.Tensor) -> torch.Tensor:
    #Gradiente de f2 evaluado en punto = (x, y).

    x = punto[0]
    y = punto[1]
    #Sacar tanto el x como el y del punto

    df2_dx = 4 * x * (x**2 + y - 11) + 2 * (x + y**2 - 7)
    df2_dy = 2 * (x**2 + y - 11) + 4 * y * (x + y**2 - 7)
    #Realiza el calculo de la derivada

    return torch.tensor([df2_dx, df2_dy])


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




if __name__ == "__main__":
    p = torch.tensor([1.0, 1.0])
    print("grad f1(1,1):", gradiente_f1(p))
    print("grad f2(1,1):", gradiente_f2(p))
