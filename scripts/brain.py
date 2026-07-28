from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap

BASE_DIR = Path(__file__).resolve().parent
NC_PATH = BASE_DIR / "PaleoDust_all.nc"

MAIN_DIR = BASE_DIR / "main"
MAIN_HOL_PATH = MAIN_DIR / "main_HOL.txt"
MAIN_LGM_PATH = MAIN_DIR / "main_LGM.txt"

# -----------------------------------------------------------------------------
# Funciones
# -----------------------------------------------------------------------------

SECONDS_PER_YEAR = 365.25 * 24 * 3600
G_M2_YR_TO_KG_M2_S = 1e-3 / SECONDS_PER_YEAR
LOG10_G_M2_YR_TO_LOG10_KG_M2_S = np.log10(G_M2_YR_TO_KG_M2_S)

def area(lat1, lon1, lat2, lon2, R=6371e3):
    #Area de una seccion de esfera.
    lat1, lat2 = np.radians([lat1, lat2])
    lon1, lon2 = np.radians([lon1, lon2])
    return R**2 * abs(lon2 - lon1) * abs(np.sin(lat2) - np.sin(lat1))


def build_area_grid(nlat, nlon):
    lat_edges = np.linspace(-90, 90, nlat + 1)
    lon_edges = np.linspace(-180, 180, nlon + 1)

    A = np.zeros((nlat, nlon), dtype=float)
    for i in range(nlat):
        for j in range(nlon):
            A[i, j] = area(lat_edges[i], lon_edges[j], lat_edges[i + 1], lon_edges[j + 1])

    return A, lat_edges, lon_edges

def global_mean_uncertainty(subds, sigma_var_name="sigma_log10_media"):

    sigma = subds[sigma_var_name].values.astype(float)

    # Se ignoran NaN o infinitos
    mask = np.isfinite(sigma)

    return np.mean(sigma[mask])

def log10_g_m2_yr_to_log10_kg_m2_s(Flog_g_m2_yr):
    return Flog_g_m2_yr + np.log10(G_M2_YR_TO_KG_M2_S)

def backtransform_lognormal_log10(mu_log10, sigma_log10):
    """
    Backtransform de una variable X positiva cuando log10(X) ~ N(mu, sigma^2).

    Devuelve:
    - mean: E[X], en las mismas unidades físicas asociadas a mu_log10
    - std: desviación estándar de X, en las mismas unidades físicas asociadas a mu_log10
    """
    sigma2_ln = (np.log(10) ** 2) * sigma_log10**2
    mean = 10**mu_log10 * np.exp(0.5 * sigma2_ln)
    std = mean * np.sqrt(np.exp(sigma2_ln) - 1.0)
    return mean, std

def get_flux_si(subds):
    """
    Convierte DMAR_log10 desde log10(g m-2 a-1)
    a log10(kg m-2 s-1) y luego hace el backtransform.
    """
    mu_si = subds["DMAR_log10"].values + LOG10_G_M2_YR_TO_LOG10_KG_M2_S
    sigma_log = subds["sigma_log10_media"].values
    return backtransform_lognormal_log10(mu_si, sigma_log)

def cargar_data_points(periodo):
    paths = {
        "HOL": MAIN_HOL_PATH,
        "LGM": MAIN_LGM_PATH,
    }

    periodo = periodo.upper()

    if periodo not in paths:
        raise ValueError("periodo debe ser 'HOL' o 'LGM'")

    path = paths[periodo]

    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    df = pd.read_csv(path, sep="\t")

    lat_col = "lat_N" if "lat_N" in df.columns else "lat"
    lon_col = "lon_E" if "lon_E" in df.columns else "lon"

    lats = df[lat_col].astype(float).values
    lons = df[lon_col].astype(float).values
    lons = ((lons + 180) % 360) - 180

    return lons, lats


# -----------------------------------------------------------------------------
# Metodo manual: datos NO backtransformed
# -----------------------------------------------------------------------------

def compute_results_manual(subds):
    """
    Calcula flujo integrado a partir de dos campos log10:
    - DMAR_log10: campo interpolado
    - sigma_log10_media: incerteza 1-sigma promedio en escala log10
    """
    F, sigma_F = get_flux_si(subds)

    nlat, nlon = F.shape
    A, lat_edges, lon_edges = build_area_grid(nlat, nlon)
    area_total = A.sum()

    M = np.sum(F * A)

    # Propagacion independiente por celda de la desviacion estandar fisica.
    # Como ahora solo tenemos un campo sigma, la cota es simetrica.
    S = np.sqrt(np.sum((sigma_F * A) ** 2))

    M_mean = M / area_total
    S_mean = S / area_total

    return {
        "M": M,
        "S": S,
        "M_mean": M_mean,
        "S_mean": S_mean,
        "lat_edges": lat_edges,
        "lon_edges": lon_edges,
        "F": F,
        "sigma_F": sigma_F,
    }


def compute_ratio_manual(res_lgm, res_hol):
    R = res_lgm["M"] / res_hol["M"]
    R_sigma = R * np.sqrt(
        (res_lgm["S"] / res_lgm["M"]) ** 2
        + (res_hol["S"] / res_hol["M"]) ** 2
    )

    return {
        "R": R,
        "R_sigma": R_sigma,
    }


# -----------------------------------------------------------------------------
# Mapas
# -----------------------------------------------------------------------------

cmap_interpolacion = plt.get_cmap("turbo")

cmap_error = LinearSegmentedColormap.from_list(
    "error_rojo",
    ["#ffffff", "#ffb3b3", "#ff4d4d", "#cc0000", "#660000"],
)

def graficar_mapa(
    datos,
    lat_edges,
    lon_edges,
    titulo,
    vmin,
    vmax,
    tipo_mapa="interpolacion",
    carpeta_salida="figures",
    mostrar_contornos=False,
    n_contornos=10,
    error=None,
    incerteza_media=None,
    mostrar_tachado=True,
    data_lons=None,
    data_lats=None,
    mostrar_data_points=False,
):
    os.makedirs(carpeta_salida, exist_ok=True)

    if tipo_mapa.lower() == "interpolacion":
        cmap = cmap_interpolacion
        etiqueta = r"Dust deposition $\log_{10}(kg\,m^{-2}\,s^{-1})$"
    elif tipo_mapa.lower() == "sigma":
        cmap = cmap_error
        etiqueta = r"Uncertainty $\sigma_{\log_{10}}$"
    else:
        raise ValueError("tipo_mapa debe ser 'interpolacion' o 'sigma'")

    vmin = float(vmin)
    vmax = float(vmax)

    if vmin > vmax:
        vmin, vmax = vmax, vmin

    n_colores = 10
    niveles_color = np.linspace(vmin, vmax, n_colores + 1)

    cmap_discreto = ListedColormap(
        cmap(np.linspace(0, 1, n_colores))
    )

    norm_discreto = BoundaryNorm(
        niveles_color,
        ncolors=cmap_discreto.N,
        clip=True
    )

    fig = plt.figure(figsize=(13, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent([-180, 180, -90, 90])
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    mapa = ax.pcolormesh(
        lon_edges,
        lat_edges,
        datos,
        cmap=cmap_discreto,
        norm=norm_discreto,
        shading="auto",
        transform=ccrs.PlateCarree(),
    )

    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    # Zonas tachadas
    if (
        mostrar_tachado
        and error is not None
        and incerteza_media is not None
    ):
        mask_tachado = np.isfinite(error) & (error > incerteza_media)

        hach = ax.contourf(
            lon_centers,
            lat_centers,
            mask_tachado.astype(float),
            levels=[0.5, 1.5],
            colors="none",
            hatches=["xxx"],
            transform=ccrs.PlateCarree(),
            zorder=5,
        )

        try:
            for coll in hach.collections:
                coll.set_edgecolor("0.3")
                coll.set_linewidth(0.2)
        except AttributeError:
            hach.set_edgecolor("0.3")
            hach.set_linewidth(0.2)

    # Data points
    if (
        mostrar_data_points
        and data_lons is not None
        and data_lats is not None
    ):
        marker_data = "+" if tipo_mapa.lower() == "sigma" else "x"

        ax.scatter(
            data_lons,
            data_lats,
            marker=marker_data,
            s=28,
            linewidths=0.8,
            color="black",
            transform=ccrs.PlateCarree(),
            zorder=7,
        )

    cbar = plt.colorbar(
        mapa,
        ax=ax,
        pad=0.03,
        boundaries=niveles_color,
        ticks=niveles_color,
        spacing="proportional"
    )
    cbar.set_label(etiqueta)

    ax.set_title(titulo)

    nombre_archivo = titulo.replace(" ", "_") + ".png"
    ruta_guardado = os.path.join(carpeta_salida, nombre_archivo)
    plt.savefig(ruta_guardado, dpi=400, bbox_inches="tight")
    print("Mapa guardado en:", ruta_guardado)
    plt.show()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def choose_variable():
    while True:
        opcion = input("¿Qué variable desea analizar?\n1) Flujo total\n2) PM10\nOpción: ")

        if opcion == "1":
            return "bulk", "total dust flux"
        if opcion == "2":
            return "pm10", "flux of PM10 particles"
        print("Por favor elige 1 o 2.\n")


def print_results_manual(
    etiqueta,
    res_hol,
    res_lgm,
    res_ratio,
    sigma_mean_hol=None,
    sigma_mean_lgm=None,
):
    print("\n" + "=" * 80)
    print(f"RESULTS | Integrated {etiqueta}".center(80))
    print("=" * 80)
    print(
        f"HOL : {res_hol['M']:.6e} kg s^-1  "
        f"± {res_hol['S']:.6e} kg s^-1"
    )
    print(
        f"LGM : {res_lgm['M']:.6e} kg s^-1  "
        f"± {res_lgm['S']:.6e} kg s^-1"
    )
    print(f"Ratio LGM/HOL : {res_ratio['R']:.4f} ± {res_ratio['R_sigma']:.4f}")
    print("=" * 80)

    print("\n" + "=" * 80)
    print(f"RESULTS | Global mean {etiqueta}".center(80))
    print("=" * 80)
    print(
        f"HOL : {res_hol['M_mean']:.6e} kg m^-2 s^-1  "
        f"± {res_hol['S_mean']:.6e} kg m^-2 s^-1"
    )
    print(
        f"LGM : {res_lgm['M_mean']:.6e} kg m^-2 s^-1  "
        f"± {res_lgm['S_mean']:.6e} kg m^-2 s^-1"
    )
    print("=" * 80)

    if sigma_mean_hol is not None and sigma_mean_lgm is not None:
        print("\n" + "=" * 80)
        print("RESULTS | Global mean interpolation uncertainty".center(80))
        print("=" * 80)
        print(f"HOL : {sigma_mean_hol:.6f} sigma_log10")
        print(f"LGM : {sigma_mean_lgm:.6f} sigma_log10")
        print("=" * 80)


def main():
    ds = xr.open_dataset(NC_PATH)

    var, etiqueta = choose_variable()

    periodos = {
        "HOL": ds.sel(period="HOL", variable=var),
        "LGM": ds.sel(period="LGM", variable=var),
    }

    resultados = {
        periodo: compute_results_manual(subds)
        for periodo, subds in periodos.items()
    }

    res_ratio = compute_ratio_manual(resultados["LGM"], resultados["HOL"])

    sigma_mean = {
        periodo: global_mean_uncertainty(subds)
        for periodo, subds in periodos.items()
    }

    print_results_manual(
        etiqueta,
        resultados["HOL"],
        resultados["LGM"],
        res_ratio,
        sigma_mean_hol=sigma_mean["HOL"],
        sigma_mean_lgm=sigma_mean["LGM"],
    )

    vmin_flux = -3 + LOG10_G_M2_YR_TO_LOG10_KG_M2_S
    vmax_flux =  3 + LOG10_G_M2_YR_TO_LOG10_KG_M2_S

    for periodo, subds in periodos.items():
        res = resultados[periodo]

        Flog = subds["DMAR_log10"].values + LOG10_G_M2_YR_TO_LOG10_KG_M2_S
        Slog = subds["sigma_log10_media"].values

        lons, lats = cargar_data_points(periodo)
        sigma_periodo = sigma_mean[periodo]

        graficar_mapa(
            Flog,
            res["lat_edges"],
            res["lon_edges"],
            f"Interpolation field of {etiqueta} {periodo}",
            vmin_flux,
            vmax_flux,
            tipo_mapa="interpolacion",
            error=Slog,
            incerteza_media=sigma_periodo,
        )

        graficar_mapa(
            Slog,
            res["lat_edges"],
            res["lon_edges"],
            f"Uncertainty field of {etiqueta} {periodo}",
            0,
            1,
            tipo_mapa="sigma",
            data_lons=lons,
            data_lats=lats,
            mostrar_data_points=True,
        )


if __name__ == "__main__":
    main()