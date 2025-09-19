import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 100)
import matplotlib.pyplot as plt


def generar_dataset(n=400, random_state=42):
    np.random.seed(random_state)

    conversion_factor = 10.7639

    data = []

    for i in range(n):
        # Primero se elige la ciudad (círculo)
        ciudad = np.random.choice(["NY", "Quilmes"], p=[0.4, 0.6])

        if ciudad == "NY":
            center_lat, center_lon = 40.7128, -74.0060
            r_max = 0.05
            max_ground_price = 10000
            decay = 0.05
            threshold = 4500
            slope = 0.1
            B_depto = 1
            C_depto = 30

            B_casa = 1.33
            C_casa = 25
        else:
            # Parámetros para Quilmes
            center_lat, center_lon = -34.7, -58.3
            r_max = 0.05
            max_ground_price = 400  # parámetros para precio según distance
            decay = 0.05  # parámetros para precio según distance
            threshold = 300  # Precio del piso donde p(depto) = 1/2
            slope = 0.1  # parámetros para p(depto)
            # Coeficientes para valor_m2
            B_depto = 1.3
            C_depto = 25
            B_casa = 1.7
            C_casa = 20

        r = r_max * np.random.uniform(0, 1)
        theta = np.random.uniform(0, 2 * np.pi)
        lat = center_lat + r * np.cos(theta)
        lon = center_lon + r * np.sin(theta)

        ground_price = max_ground_price * np.exp(-r / decay)

        p_depto = 1 / (1 + np.exp(-slope * (ground_price - threshold)))
        tipo = "depto" if np.random.uniform(0, 1) < p_depto else "casa"

        if tipo == "depto":
            edad = np.random.uniform(1, 15)
        else:
            edad = np.random.uniform(1, 50)
        if np.random.uniform(0, 1) < 0.1:
            edad_sim = np.nan
        else:
            edad_sim = edad

        if tipo == "depto":
            valor_m2 = ground_price * B_depto - C_depto * np.log(edad)
        else:
            valor_m2 = ground_price * B_casa - C_casa * np.log(edad)

        valor_m2 = np.maximum(valor_m2, 1)

        # Se simula el área en m². Se hace que a mayor precio del suelo el área sea menor.
        if tipo == "depto":
            mu_area = 80 - 50 * (ground_price / max_ground_price)
            area_m2 = np.random.normal(loc=mu_area, scale=5)
            area_m2 = max(area_m2, 20)  # se fija un mínimo
        else:
            # Para casas, se agrega además que las viviendas más viejas (log(edad)) tienen más chances de ser grandes.
            mu_area = 200 - 100 * (ground_price / max_ground_price) + 10 * np.log(edad)
            area_m2 = np.random.normal(loc=mu_area, scale=20)
            area_m2 = max(area_m2, 30)

        # Se determina la cantidad de ambientes en función del área
        ambientes = np.random.poisson(lam=area_m2 / 13) + 1
        ambientes = int(np.clip(ambientes, 1, 8))

        # Se asigna la cantidad de pisos:
        # Para casas: si ambientes > 4 => 2 pisos; si > 6 => 3 pisos; caso contrario 1 piso.
        # Para departamentos se deja como NaN.
        if tipo == "casa":
            if ambientes > 6:
                pisos = 3
            elif ambientes > 4:
                pisos = 2
            else:
                pisos = 1
        else:
            pisos = np.nan

        # Se simula la existencia de pileta (más probable en casas)
        if tipo == "casa":
            p_pileta = 0.4
        else:
            p_pileta = 0.005
        tiene_pileta = np.random.uniform(0, 1) < p_pileta

        # Se calcula el precio final: precio = área (en m²) * valor_m2.
        # Si tiene pileta se incrementa el precio en un 10%.
        precio = area_m2 * valor_m2
        if tiene_pileta:
            precio *= 1.1
        precio *= np.random.normal(loc=1, scale=0.17)  # Add noise (20% std dev)

        # Se informa el área en unidades distintas según la ciudad:
        if ciudad == "NY":
            area_reported = area_m2 * conversion_factor  # convertir a sqft
            unidades = "sqft"
        else:
            area_reported = area_m2
            unidades = "m2"

        # Se arma la fila (la mayoría de las “variables escondidas” no se muestran en el dataset final)
        row = {
            "precio": precio,
            "tipo": tipo,
            "Área": area_reported,
            "unidades": unidades,
            "ambientes": ambientes,
            "pisos": pisos,
            "pileta": tiene_pileta,
            "lat": lat,
            "lon": lon,
            "edad": edad_sim,
        }
        data.append(row)

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    df = generar_dataset(n=415, random_state=42)
    print("Primeras filas del dataset generado:")
    print(df.head())

    # Split into train (80%) and test (20%)
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(0.8 * len(df_shuffled))
    df_train = df_shuffled.iloc[:split_idx]
    df_test = df_shuffled.iloc[split_idx:]

    # Save to CSV
    df_train.to_csv("train.csv", index=False)
    df_test.to_csv("test.csv", index=False)
    print(f"Train set saved to train.csv ({len(df_train)} rows)")
    print(f"Test set saved to test.csv ({len(df_test)} rows)")

    plt.hist(df["precio"][df["precio"] < 1000000], 40)
    plt.show()
