from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "csv"

# True para obtener un archivo unico con toda la info
CREATE_COMBINED_FILE = True
COMBINED_FILENAME = "PaleoDust_all.nc"


# -----------------------------------------------------------------------------
# Mapeos de nombres
# -----------------------------------------------------------------------------

# Se acepta FLUX como alias de BULK, pero dentro del NetCDF queda como "bulk"
VAR_NAME_MAP = {
    "BULK": "bulk",
    "FLUX": "bulk",
    "PM10": "pm10",
}

FIELD_NAME_MAP_MANUAL = {
    "IF": "DMAR_log10",
    "1S": "sigma_log10_media",
}

LONG_NAME_MAP_MANUAL = {
    "DMAR_log10": "Interpolated DMAR field (log10 scale)",
    "sigma_log10_media": "Average 1-sigma uncertainty (log10 scale)",
}

UNITS_MAP_MANUAL = {
    "DMAR_log10": "log10(g m-2 a-1)",
    "sigma_log10_media": "log10(g m-2 a-1)",
}

# Modo manual:
# HOL_FLUX_IF.csv, HOL_FLUX_1S.csv, HOL_PM10_IF.csv, etc.
FILENAME_RE_MANUAL = re.compile(
    r"^(HOL|LGM)_(BULK|FLUX|PM10)_(IF|1S)\.csv$",
    re.IGNORECASE,
)


def normalize_variable(variable: str) -> str:
    # Usa BULK internamente para agrupar FLUX/BULK como el mismo campo
    variable = variable.upper()
    if variable == "FLUX":
        return "BULK"
    return variable


def get_mode_config():
    return {
        "field_name_map": FIELD_NAME_MAP_MANUAL,
        "long_name_map": LONG_NAME_MAP_MANUAL,
        "units_map": UNITS_MAP_MANUAL,
        "filename_re": FILENAME_RE_MANUAL,
        "method": "Kriging with log10 transformation",
        "comment": (
            "Includes only the interpolated log10 DMAR field and the average "
            "1-sigma uncertainty field in log10 scale. Backtransform is done "
            "later in the analysis script."
        ),
        "missing_example": "Ejemplo: HOL_FLUX_IF.csv, HOL_FLUX_1S.csv.",
    }


# -----------------------------------------------------------------------------
# Funciones
# -----------------------------------------------------------------------------

def discover_files(data_dir: Path) -> Dict[Tuple[str, str], Dict[str, Path]]:
    config = get_mode_config()
    filename_re = config["filename_re"]

    grouped: Dict[Tuple[str, str], Dict[str, Path]] = {}

    for path in data_dir.glob("*.csv"):
        match = filename_re.match(path.name)
        if not match:
            continue

        period, variable, field_code = match.groups()
        key = (period.upper(), normalize_variable(variable))
        grouped.setdefault(key, {})[field_code.upper()] = path

    return grouped


def read_csv_matrix(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None).to_numpy(dtype=float)


def build_coords(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    nlat, nlon = shape
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    return lat, lon


def build_dataset(period: str, variable: str, file_map: Dict[str, Path]) -> xr.Dataset:
    config = get_mode_config()
    field_name_map = config["field_name_map"]
    long_name_map = config["long_name_map"]
    units_map = config["units_map"]

    required = set(field_name_map.keys())
    missing = required - set(file_map.keys())
    if missing:
        raise ValueError(
            f"Faltan archivos para {period}_{variable}: {sorted(missing)}"
        )

    arrays = {code: read_csv_matrix(path) for code, path in file_map.items()}

    shapes = {arr.shape for arr in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(
            f"Las dimensiones no coinciden para {period}_{variable}: {shapes}"
        )

    shape = next(iter(shapes))
    lat, lon = build_coords(shape)

    data_vars = {}
    for field_code in sorted(required):
        var_name = field_name_map[field_code]
        data_vars[var_name] = (("lat", "lon"), arrays[field_code])

    ds = xr.Dataset(data_vars=data_vars, coords={"lat": lat, "lon": lon})

    ds["lat"].attrs = {"units": "degrees_north", "standard_name": "latitude"}
    ds["lon"].attrs = {"units": "degrees_east", "standard_name": "longitude"}

    for var_name in ds.data_vars:
        ds[var_name].attrs = {
            "units": units_map[var_name],
            "long_name": long_name_map[var_name],
        }

    ds.attrs = {
        "title": f"{period} {variable} Dust Field",
        "source": "Global-Kriger Toolbox",
        "method": config["method"],
        "period": period,
        "variable": VAR_NAME_MAP[variable],
        "author": "Bruno Varela Molina",
        "history": "Created using Python xarray in WSL",
        "comment": config["comment"],
    }

    return ds


def save_individual_files(
    grouped_files: Dict[Tuple[str, str], Dict[str, Path]],
    base_dir: Path,
) -> Dict[Tuple[str, str], Path]:
    outputs: Dict[Tuple[str, str], Path] = {}

    for (period, variable), file_map in sorted(grouped_files.items()):
        ds = build_dataset(period, variable, file_map)
        output_path = base_dir / f"{period}_{variable}.nc"
        ds.to_netcdf(output_path)
        outputs[(period, variable)] = output_path
        print(f"NetCDF creado: {output_path.name}")

    return outputs


def save_combined_file(
    grouped_files: Dict[Tuple[str, str], Dict[str, Path]],
    base_dir: Path,
    filename: str,
) -> Path:
    datasets = []

    for (period, variable), file_map in sorted(grouped_files.items()):
        ds = build_dataset(period, variable, file_map)
        ds = ds.expand_dims(period=[period], variable=[VAR_NAME_MAP[variable]])
        datasets.append(ds)

    combined = xr.combine_by_coords(datasets, combine_attrs="override")
    combined.attrs = {
        "title": "PaleoDust combined dataset",
        "source": "Global-Kriger Toolbox",
        "method": get_mode_config()["method"],
        "author": "Bruno Varela Molina",
        "history": "Combined using Python xarray in WSL",
        "comment": "Dimensions: period x variable x lat x lon.",
    }

    output_path = base_dir / filename
    combined.to_netcdf(output_path)
    print(f"NetCDF combinado creado: {output_path.name}")

    return output_path


def main() -> None:
    grouped_files = discover_files(DATA_DIR)

    if not grouped_files:
        raise FileNotFoundError(
            "No se encontraron archivos CSV con el patrón esperado. "
            + get_mode_config()["missing_example"]
        )

    expected_groups = {
        ("HOL", "BULK"),
        ("HOL", "PM10"),
        ("LGM", "BULK"),
        ("LGM", "PM10"),
    }

    missing_groups = expected_groups - set(grouped_files.keys())
    if missing_groups:
        raise ValueError(
            f"Faltan grupos completos de archivos: {sorted(missing_groups)}"
        )

    save_individual_files(grouped_files, BASE_DIR)

    if CREATE_COMBINED_FILE:
        save_combined_file(grouped_files, BASE_DIR, COMBINED_FILENAME)


if __name__ == "__main__":
    main()