import os
import pandas as pd
import numpy as np
import openpyxl
import time

ruta = r"C:\Users\Mellizos\OneDrive\Área de Trabalho\CADOI\Resumenes\PaleoDust Resumen 7\main_HOL.txt"

df = pd.read_csv(ruta, sep='\t')
df = df[(df["DMAR"] != 0) | (df["DMAR10"] != 0)]
df = df.reset_index(drop=True)
N = len(df)

num_cols = ["lon", "lat", "DMAR", "sigma", "DMAR10", "sigma10"]
df[num_cols] = df[num_cols].apply(pd.to_numeric)

###################################|###################################
##### DEFINICION DE FUNCIONES #####|##### DEFINICION DE FUNCIONES #####
###################################|###################################

def distance(lon1, lat1, lon2, lat2):
    R = 6371.0 #Radio de la Tierra en km
    
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c # distancia en km

###################################|###################################
##### CONSTRUCCION DE CLUSTERS ####|#### CONSTRUCCION DE CLUSTERS #####
###################################|###################################

R = 11.11 #0.1 grados de lat/lon en el ecuador
t0 = time.perf_counter()

#Creamos el diccionario de vecinos, donde la idea es
    #"el punto i tiene como vecinos a los puntos []"

vecinos = {i: [] for i in range(N)}

for i in range(N):
    for j in range(i + 1, N):
        dist = distance(
            df.loc[i]['lon'], df.loc[i]['lat'],
            df.loc[j]['lon'], df.loc[j]['lat'],
            )
        if dist <= R:
            vecinos[i].append(j)
            vecinos[j].append(i)

visitado = set()
clusters = []

for i in range(N):
    if i in visitado:
        continue

    cluster = []
    stack   = [i]

    while stack:
        k = stack.pop()
        if k in visitado:
            continue
        visitado.add(k)
        cluster.append(k)
        stack.extend(vecinos[k])

    clusters.append(cluster)

###################################|###################################
######## PRINT DE CLUSTERS ########|######## (NO OBLIGATORIO!) ########
###################################|###################################

for n, c in enumerate(clusters, start=1):
#    print(f"\n-| Cluster N°: {n} |-  (size={len(c)})")

#    # Sub-DataFrame con las filas del cluster
#    sub = df.loc[c, ["lon", "lat", "DMAR", "sigma", "DMAR10", "sigma10", "type_core"]]

#    # opcional: ordenar por lat/lon para que se vea más coherente
#    sub = sub.sort_values(["lat", "lon"]).reset_index().rename(columns={"index": "id"})

#    print(sub.to_string(index=False))
    total = n
print(f"Para {R} km se obtuvieron {n} clusters")
t1 = time.perf_counter()
print(f"Tiempo de cálculo: {t1 - t0:.2f} segundos")


registros = []
registros_10 = []

for cluster in clusters:
    lon_mean   = df.loc[cluster, 'lon'].mean()
    lat_mean   = df.loc[cluster, 'lat'].mean()
    DMAR_mean  = df.loc[cluster, 'DMAR'].mean()
    DMAR10_mean  = df.loc[cluster, 'DMAR10'].mean()
    
    sig   = df.loc[cluster, "sigma"].to_numpy()
    sig10 = df.loc[cluster, "sigma10"].to_numpy()
    n     = sig.size
    n10   = sig10.size

    sigma_mean   = np.sqrt(np.sum(sig**2)) / n
    sigma10_mean = np.sqrt(np.sum(sig10**2)) / n10

    serie = df.loc[cluster, 'type_core'].dropna()
    type_core = serie.mode().iloc[0] if len(serie) > 0 else "NaN"

    registros.append({
        "lon"     : lon_mean,
        "lat"     : lat_mean,
        "DMAR"    : DMAR_mean,
        "sigma"   : sigma_mean * 0.7,
        "type"    : type_core
        })

    registros_10.append({
        "lon"     : lon_mean,
        "lat"     : lat_mean,
        "DMAR10"  : DMAR10_mean,
        "sigma10" : sigma10_mean * 0.7,
        "type"    : type_core
        })

df_reg = pd.DataFrame(registros)
df_reg10 = pd.DataFrame(registros_10)

carpeta = os.path.dirname(ruta)

ruta_excel     = os.path.join(carpeta, "HOL_FLUX.xlsx")
ruta_excel_10  = os.path.join(carpeta, "HOL_PM10.xlsx")

df_reg.to_excel(ruta_excel, index=False)
df_reg10.to_excel(ruta_excel_10, index=False)
