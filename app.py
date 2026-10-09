import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

# Configuración de la página web de Streamlit
st.set_page_config(
    page_title="Despliegue - Predicción de Sentimientos de Reviews",
    page_icon="📊",
    layout="wide"
)

# Título y Descripción de la Aplicación
st.title("🧠 Clasificador de Sentimiento de Reviews")
st.markdown("""
Esta aplicación web dinámica utiliza un modelo **Soporte Vectorial (SVM)** optimizado con balanceo SMOTE en el entrenamiento para predecir si el sentimiento de un comentario es **Negativo, Neutro o Positivo**.

Puedes ingresar los datos de forma **individual** en la barra lateral o cargar un **archivo de datos (CSV o Excel)** en el panel central para realizar predicciones masivas en tiempo real.
""")

# --- 1. DEFINICIÓN DE CONFIGURACIÓN POR DEFECTO Y CARGA DE MODELO ---
# Valores promedio de entrenamiento para imputación de nulos si el usuario sube datos incompletos
MEANS_ENTRENAMIENTO = {
    'exclamation_count': 2.4225,
    'positive_word_score': 0.5180,
    'negative_word_score': 0.4739,
    'neutral_word_score': 0.5331
}

FEATURES = ['exclamation_count', 'positive_word_score', 'negative_word_score', 'neutral_word_score']
MAP_SENTIMENT = {0: 'negative', 1: 'neutral', 2: 'positive'}

@st.cache_resource
def cargar_artefactos():
    """Carga los artefactos del modelo de forma segura."""
    try:
        # Intenta cargar los modelos guardados localmente
        pipeline = joblib.load('svm_sentiment_pipeline.joblib')
        return pipeline
    except Exception:
        return None

model_pipeline = cargar_artefactos()

if model_pipeline is None:
    st.warning("⚠️ No se encontraron los archivos locales del modelo entrenado (`svm_sentiment_pipeline.joblib`). Se simulará la respuesta del modelo SVM entrenado utilizando umbrales lógicos de las métricas para que la app sea funcional de manera autónoma.")

def predecir(df_input):
    """Aplica el modelo o lógica umbral para obtener predicciones."""
    # 1. Asegurar la presencia de las características requeridas y rellenar nulos
    df_proc = df_input.copy()
    for col in FEATURES:
        if col not in df_proc.columns:
            df_proc[col] = MEANS_ENTRENAMIENTO[col]
        else:
            df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce').fillna(MEANS_ENTRENAMIENTO[col])
    
    X_features = df_proc[FEATURES]
    
    # 2. Realizar predicción
    if model_pipeline is not None:
        preds = model_pipeline.predict(X_features)
        # Obtener probabilidades si están disponibles
        try:
            probs = model_pipeline.predict_proba(X_features)
            prob_max = np.max(probs, axis=1)
        except:
            prob_max = [1.0] * len(preds)
        
        resultados = [MAP_SENTIMENT[p] for p in preds]
        confianzas = [f"{p*100:.1f}%" for p in prob_max]
    else:
        # Lógica de respaldo robusta que imita el comportamiento del modelo entrenado
        resultados = []
        confianzas = []
        for _, row in X_features.iterrows():
            pos = row['positive_word_score']
            neg = row['negative_word_score']
            neu = row['neutral_word_score']
            
            if neg > pos and neg > neu:
                resultados.append('negative')
                confianzas.append(f"{max(70.0, neg*100):.1f}%")
            elif pos > neg and pos > neu:
                resultados.append('positive')
                confianzas.append(f"{max(70.0, pos*100):.1f}%")
            else:
                resultados.append('neutral')
                confianzas.append(f"{max(70.0, neu*100):.1f}%")
                
    df_proc['sentimiento_predicho'] = resultados
    df_proc['confianza_prediccion'] = confianzas
    return df_proc

# --- 2. INTERFAZ: PREDICCIÓN INDIVIDUAL (BARRA LATERAL) ---
st.sidebar.header("🔍 Predicción Individual Individual")
with st.sidebar.form("individual_form"):
    input_exc = st.slider("Cantidad de Exclamaciones (exclamation_count)", 0, 20, 2)
    input_pos = st.slider("Score de Palabras Positivas (positive_word_score)", 0.0, 1.0, 0.5, step=0.01)
    input_neg = st.slider("Score de Palabras Negativas (negative_word_score)", 0.0, 1.0, 0.3, step=0.01)
    input_neu = st.slider("Score de Palabras Neutras (neutral_word_score)", 0.0, 1.0, 0.5, step=0.01)
    
    submit_btn = st.form_submit_button("Predecir Sentimiento")

if submit_btn:
    data_ind = pd.DataFrame([{
        'exclamation_count': input_exc,
        'positive_word_score': input_pos,
        'negative_word_score': input_neg,
        
        'neutral_word_score': input_neu
    }])
    
    res_df = predecir(data_ind)
    pred_sent = res_df.loc[0, 'sentimiento_predicho']
    pred_conf = res_df.loc[0, 'confianza_prediccion']
    
    st.sidebar.markdown("--- ")
    st.sidebar.subheader("Resultado de Predicción:")
    if pred_sent == 'positive':
        st.sidebar.success(f"🟢 **Positivo** (Confianza: {pred_conf})")
    elif pred_sent == 'negative':
        st.sidebar.error(f"🔴 **Negativo** (Confianza: {pred_conf})")
    else:
        st.sidebar.warning(f"🟡 **Neutro** (Confianza: {pred_conf})")

# --- 3. INTERFAZ: ENTORNO DINÁMICO DE CARGA MASIVA (PANEL CENTRAL) ---
st.subheader("📂 Carga de base de datos a predecir")
st.write("Sube un archivo en formato **CSV o Excel** que contenga registros a evaluar. El sistema detectará las variables necesarias o imputará valores faltantes de manera inteligente.")

archivo_subido = st.file_uploader("Selecciona el archivo (.csv, .xlsx, .xls)", type=["csv", "xlsx", "xls"])

if archivo_subido is not None:
    try:
        # Lectura dinámica del formato de archivo
        nombre_archivo = archivo_subido.name
        if nombre_archivo.endswith('.csv'):
            df_cargado = pd.read_csv(archivo_subido)
        else:
            df_cargado = pd.read_excel(archivo_subido)
            
        st.success(f"✅ Archivo '{nombre_archivo}' cargado con éxito. Se detectaron **{len(df_cargado)} registros**.")
        
        # Mostrar vista previa de los datos subidos
        with st.expander("👁️ Ver primeros registros del archivo subido"):
            st.dataframe(df_cargado.head(10))
            
        # Procesamiento y Predicción
        with st.spinner("🤖 Procesando y calculando sentimientos..."):
            df_predicho = predecir(df_cargado)
            
        st.subheader("🎯 Resultados de las Predicciones")
        
        # Resumen gráfico en la app
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Distribución de sentimientos:**")
            resumen_sent = df_predicho['sentimiento_predicho'].value_counts()
            st.dataframe(resumen_sent)
        with col2:
            # Vista del dataframe final con las nuevas columnas al inicio
            cols_result = ['sentimiento_predicho', 'confianza_prediccion'] + [c for c in df_predicho.columns if c not in ['sentimiento_predicho', 'confianza_prediccion']]
            st.markdown("**Tabla de resultados con predicciones:**")
            st.dataframe(df_predicho[cols_result].head(100))
            
        # Botones de descarga dinámica sin dependencias del servidor
        st.markdown("### 📥 Descargar Resultados")
        st.write("Descarga la base de datos procesada con las columnas de predicción incorporadas de forma local.")
        
        col_btn1, col_btn2 = st.columns(2)
        
        # Descarga Excel
        buffer_excel = io.BytesIO()
        with pd.ExcelWriter(buffer_excel, engine='xlsxwriter') as writer:
            df_predicho.to_excel(writer, index=False, sheet_name='Predicciones')
        
        col_btn1.download_button(
            label="📥 Descargar en Excel (.xlsx)",
            data=buffer_excel.getvalue(),
            file_name="predicciones_sentimiento.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Descarga CSV
        csv_data = df_predicho.to_csv(index=False).encode('utf-8')
        col_btn2.download_button(
            label="📥 Descargar en CSV (.csv)",
            data=csv_data,
            file_name="predicciones_sentimiento.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"❌ Ocurrió un error al procesar el archivo: {e}")
else:
    # Mostrar una guía visual si no se ha cargado ningún archivo
    st.info("💡 **Consejo:** Sube tu archivo con las columnas `exclamation_count`, `positive_word_score`, `negative_word_score` y `neutral_word_score` para obtener un análisis optimizado de forma inmediata.")
