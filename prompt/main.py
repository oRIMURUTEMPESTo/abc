import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import seaborn as sns
from datetime import datetime, timedelta
import re
import streamlit as st

def cargar_archivos_excel(ruta_carpeta):
    dataframes = []
    for nombre_archivo in os.listdir(ruta_carpeta):
        if nombre_archivo.endswith('.xlsx'):
            ruta_archivo = os.path.join(ruta_carpeta, nombre_archivo)
            try:
                df = pd.read_excel(ruta_archivo, sheet_name="Mrbeast_stats")
                dataframes.append(df)
            except Exception as e:
                st.error(f"Error al cargar el archivo {nombre_archivo}: {e}")
    return dataframes

def seleccionar_columnas(dataframes):
    dataframes_seleccionados = []
    for df in dataframes:
        if set(['Views', 'Date', 'Duration']).issubset(df.columns):
            datos_seleccionados = df[['Views', 'Date', 'Duration']]
            dataframes_seleccionados.append(datos_seleccionados)
    return pd.concat(dataframes_seleccionados, ignore_index=True)

def exportar_dataset(dataset, ruta_salida):
    try:
        dataset.to_excel(ruta_salida, index=False)
        st.success(f"Dataset exportado exitosamente a {ruta_salida}")
    except Exception as e:
        st.error(f"Error al exportar el dataset: {e}")

def limpiar_datos(df):
    df.dropna(inplace=True)
    df['Views'] = df['Views'].apply(convertir_vistas)
    df['Date'] = df['Date'].apply(convertir_fecha)
    df['Duration'] = df['Duration'].apply(convertir_duracion)
    df.dropna(inplace=True)  # Eliminar filas con valores NaN después de la conversión
    df['Dias_desde_subida'] = (pd.Timestamp.now() - df['Date']).dt.days
    return df

def convertir_vistas(vistas):
    try:
        if 'M' in vistas:
            return float(vistas.replace('M views', '')) * 1e6
        elif 'K' in vistas:
            return float(vistas.replace('K views', '')) * 1e3
        return float(vistas.replace(' views', ''))
    except ValueError:
        return None

def convertir_fecha(fecha_str):
    try:
        match = re.match(r'(\d+)\s(\w+)\sago', fecha_str)
        if match:
            valor, unidad = match.groups()
            valor = int(valor)
            dias = {
                'day': 1, 'days': 1,
                'week': 7, 'weeks': 7,
                'month': 30, 'months': 30,
                'year': 365, 'years': 365
            }
            return pd.Timestamp.now() - timedelta(days=valor * dias.get(unidad, 1))
        return None
    except Exception as e:
        return None

def convertir_duracion(duracion_str):
    try:
        partes = duracion_str.split(':')
        if len(partes) == 2:
            minutos, segundos = partes
            return int(minutos) * 60 + int(segundos)
        elif len(partes) == 3:
            horas, minutos, segundos = partes
            return int(horas) * 3600 + int(minutos) * 60 + int(segundos)
        return None
    except Exception as e:
        return None

def analizar_y_modelar(df):
    X_tiempo = df[['Dias_desde_subida']]
    X_duracion = df[['Duration']]
    y = df['Views']
    
    X_train_tiempo, X_test_tiempo, y_train, y_test = train_test_split(X_tiempo, y, test_size=0.2, random_state=42)
    X_train_duracion, X_test_duracion, y_train_duracion, y_test_duracion = train_test_split(X_duracion, y, test_size=0.2, random_state=42)
    
    modelos = {
        "Regresión Lineal": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Regresión Polinómica (grado=2)": PolynomialFeatures(degree=2),
        "Regresión de Soporte Vectorial": SVR(kernel='rbf'),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "K-Nearest Neighbors": KNeighborsRegressor()
    }
    
    resultados = {"Dias_desde_subida": {}, "Duration": {}}
    
    for nombre, modelo in modelos.items():
        # Para Dias_desde_subida
        if nombre.startswith("Regresión Polinómica"):
            poly = PolynomialFeatures(degree=2)
            X_poly_tiempo = poly.fit_transform(X_train_tiempo)
            modelo_tiempo = LinearRegression().fit(X_poly_tiempo, y_train)
            X_test_poly_tiempo = poly.transform(X_test_tiempo)
            y_pred_tiempo = modelo_tiempo.predict(X_test_poly_tiempo)
        else:
            modelo.fit(X_train_tiempo, y_train)
            y_pred_tiempo = modelo.predict(X_test_tiempo)
        
        r2_tiempo = r2_score(y_test, y_pred_tiempo)
        resultados["Dias_desde_subida"][nombre] = r2_tiempo
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred_tiempo, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set_xlabel('Vistas Reales')
        ax.set_ylabel('Vistas Predichas')
        ax.set_title(f'{nombre} - Predicción de Días desde Subida (R² = {r2_tiempo:.2f})')
        ax.grid(True)
        st.pyplot(fig)
        
        # Para Duration
        if nombre.startswith("Regresión Polinómica"):
            X_poly_duracion = poly.fit_transform(X_train_duracion)
            modelo_duracion = LinearRegression().fit(X_poly_duracion, y_train_duracion)
            X_test_poly_duracion = poly.transform(X_test_duracion)
            y_pred_duracion = modelo_duracion.predict(X_test_poly_duracion)
        else:
            modelo.fit(X_train_duracion, y_train_duracion)
            y_pred_duracion = modelo.predict(X_test_duracion)
        
        r2_duracion = r2_score(y_test_duracion, y_pred_duracion)
        resultados["Duration"][nombre] = r2_duracion
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test_duracion, y_pred_duracion, alpha=0.5)
        ax.plot([y_test_duracion.min(), y_test_duracion.max()], [y_test_duracion.min(), y_test_duracion.max()], 'k--', lw=2)
        ax.set_xlabel('Vistas Reales')
        ax.set_ylabel('Vistas Predichas')
        ax.set_title(f'{nombre} - Predicción de Duración (R² = {r2_duracion:.2f})')
        ax.grid(True)
        st.pyplot(fig)
    
    mejor_modelo_tiempo = max(resultados["Dias_desde_subida"], key=resultados["Dias_desde_subida"].get)
    mejor_modelo_duracion = max(resultados["Duration"], key=resultados["Duration"].get)
    
    st.write(f"El mejor modelo para Días desde Subida es {mejor_modelo_tiempo} con un R² de {resultados['Dias_desde_subida'][mejor_modelo_tiempo]:.2f}")
    st.write(f"El mejor modelo para Duración es {mejor_modelo_duracion} con un R² de {resultados['Duration'][mejor_modelo_duracion]:.2f}")

    # Visualización de la relación entre las variables
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x=df['Dias_desde_subida'], y=df['Views'], ax=ax)
    ax.set_title('Relación entre Días desde Subida y Número de Vistas')
    ax.set_xlabel('Días desde Subida')
    ax.set_ylabel('Número de Vistas')
    ax.grid(True)
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x=df['Duration'], y=df['Views'], ax=ax)
    ax.set_title('Relación entre Duración y Número de Vistas')
    ax.set_xlabel('Duración (segundos)')
    ax.set_ylabel('Número de Vistas')
    ax.grid(True)
    st.pyplot(fig)

    # Gráfico de torta, barras y líneas
    df['Categoria_Duracion'] = pd.cut(df['Duration'], bins=[0, 600, 1200, float('inf')], labels=['Corta', 'Media', 'Larga'])
    duracion_counts = df['Categoria_Duracion'].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(duracion_counts, labels=duracion_counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title('Distribución de Duración de Videos')
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=duracion_counts.index, y=duracion_counts.values, ax=ax)
    ax.set_title('Conteo de Videos por Categoría de Duración')
    ax.set_xlabel('Categoría de Duración')
    ax.set_ylabel('Número de Videos')
    ax.grid(True)
    st.pyplot(fig)
    
    return resultados

def main():
    st.title("Análisis de Estadísticas de Videos de MrBeast")
    st.markdown("Esta aplicación permite cargar archivos Excel con estadísticas de videos de MrBeast, limpiarlos, analizarlos y modelar las vistas en función del tiempo desde la subida y la duración del video.")

    ruta_carpeta = st.text_input("Ingrese la ruta de la carpeta que contiene los archivos Excel")
    ruta_salida = st.text_input("Ingrese la ruta de salida para exportar el dataset limpio")

    if st.button("Cargar y Limpiar Datos"):
        dataframes = cargar_archivos_excel(ruta_carpeta)
        if dataframes:
            dataset = seleccionar_columnas(dataframes)
            dataset_limpio = limpiar_datos(dataset)
            exportar_dataset(dataset_limpio, ruta_salida)
            st.dataframe(dataset_limpio)
            st.session_state.dataset_limpio = dataset_limpio
        else:
            st.warning("No se encontraron archivos Excel válidos en la carpeta especificada.")
    
    if st.button("Analizar y Modelar Datos"):
        if 'dataset_limpio' in st.session_state:
            dataset_limpio = st.session_state.dataset_limpio
            resultados = analizar_y_modelar(dataset_limpio)
            st.write(resultados)
        else:
            st.warning("Primero debe cargar y limpiar los datos.")

if __name__ == "__main__":
    main()
