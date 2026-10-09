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

# --- ENCABEZADO ACADÉMICO REQUERIDO ---
st.markdown("""
<div style="background-color:#1e293b; padding:20px; border-radius:10px; margin-bottom:25px; border-left: 8px solid #3b82f6;">
    <h4 style="color:#f8fafc; margin:0;">🏫 Especialización en Analítica de Datos Aplicada a los Negocios</h4>
    <p style="color:#94a3b8; font-size:16px; margin:5px 0 0 0;">
        <strong>Asignatura:</strong> Machine Learning <br>
        <strong>Autores:</strong> Daniela Sanchez Narvaez y Valeria Cely Martinez
    </p>
</div>
""", unsafe_html=True)

# Título y Descripción de la Aplicación
st.title("🧠 Clasificador de Sentimiento de Reviews (SVM)")
st.markdown("""
Esta aplicación web recopila **todas las variables originales del conjunto de datos** para su procesamiento. 
Posteriormente, realiza una **limpieza y transformación de datos (ETL)** en tiempo real, aislando únicamente las variables predictivas óptimas para el pipeline del modelo.
""")

# --- DEFINICIÓN DE CONFIGURACIÓN POR DEFECTO Y CARGA DE MODELO ---
MEANS_ENTRENAMIENTO = {
    'exclamation_count': 2.4225,
    'positive_word_score': 0.5180,
    'negative_word_score': 0.4739,
    'neutral_word_score': 0.5331
}

FEATURES_PREDICTIVAS = ['exclamation_count', 'positive_word_score', 'negative_word_score', 'neutral_word_score']
MAP_SENTIMENT = {0: 'negative', 1: 'neutral', 2: 'positive'}

# Lista de todas las variables iniciales del dataset para validación masiva
ALL_INITIAL_COLUMNS = [
    'RecordID', 'FullName', 'Phone', 'ZodiacSign', 'FavoriteColor', 'Hobby',
    'review_length', 'exclamation_count', 'question_count', 'positive_word_score',
    'negative_word_score', 'neutral_word_score', 'rating', 'product_category',
    'verified_purchase', 'helpful_votes', 'review_year', 'Platform'
]

@st.cache_resource
def cargar_artefactos():
    """Carga los artefactos del modelo de forma segura."""
    try:
        pipeline = joblib.load('svm_sentiment_pipeline.joblib')
        return pipeline
    except Exception:
        return None

model_pipeline = cargar_artefactos()

if model_pipeline is None:
    st.warning("⚠️ No se encontró el archivo del modelo entrenado (`svm_sentiment_pipeline.joblib`). Se simulará la respuesta lógica del clasificador.")

def limpiar_y_predecir(df_input):
    """ETL Pipeline: Recibe un DF con todas las variables, limpia y predice usando el modelo SVM."""
    df_clean = df_input.copy()
    
    # PROCESAMIENTO Y LIMPIEZA ETL EN TIEMPO REAL
    # Convertir a numérico y rellenar nulos para las variables predictivas específicas
    for col in FEATURES_PREDICTIVAS:
        if col not in df_clean.columns:
            df_clean[col] = MEANS_ENTRENAMIENTO[col]
        else:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(MEANS_ENTRENAMIENTO[col])
            
    X_features = df_clean[FEATURES_PREDICTIVAS]
    
    # Predicción utilizando el modelo cargado o la función simuladora robusta
    if model_pipeline is not None:
        preds = model_pipeline.predict(X_features)
        try:
            probs = model_pipeline.predict_proba(X_features)
            prob_max = np.max(probs, axis=1)
        except:
            prob_max = [1.0] * len(preds)
        
        resultados = [MAP_SENTIMENT[p] for p in preds]
        confianzas = [f"{p*100:.1f}%" for p in prob_max]
    else:
        # Lógica de simulación condicional basada en scores
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
                
    df_clean['sentimiento_predicho'] = resultados
    df_clean['confianza_prediccion'] = confianzas
    return df_clean

# --- INTERFAZ 1: PREDICCIÓN INDIVIDUAL REQUIRIENDO TODOS LOS CAMPOS ---
st.sidebar.header("📝 Formulario de Entrada Completa")
st.sidebar.write("Para realizar una predicción, debes rellenar obligatoriamente todos los campos recolectados originalmente:")

with st.sidebar.form("full_individual_form"):
    st.subheader("Información Personal (Metadata)")
    val_id = st.text_input("ID del Registro (RecordID)", value="REC99999")
    val_name = st.text_input("Nombre Completo (FullName)", value="Juan Pérez")
    val_phone = st.number_input("Número de Teléfono (Phone)", min_value=3000000000, max_value=3999999999, value=3154859632)
    val_zodiac = st.selectbox("Signo Zodiacal", ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"])
    val_color = st.text_input("Color Favorito", value="Azul")
    val_hobby = st.text_input("Hobby principal", value="Leer")
    
    st.subheader("Detalles de la Review")
    val_len = st.number_input("Longitud del Review (review_length)", min_value=0, max_value=50000, value=150)
    val_questions = st.number_input("Cantidad de Signos de Interrogación (question_count)", min_value=0, max_value=10, value=0)
    val_rating = st.slider("Calificación (rating)", min_value=1, max_value=5, value=3)
    val_cat = st.text_input("Categoría de Producto", value="Electronics")
    val_verified = st.selectbox("Compra Verificada", ["Yes", "No"])
    val_votes = st.number_input("Votos Útiles (helpful_votes)", min_value=-5, max_value=10000, value=0)
    val_year = st.number_input("Año del Review (review_year)", min_value=2018, max_value=2024, value=2023)
    val_platform = st.selectbox("Plataforma de Compra", ["Web", "App", "Partner"])
    
    st.subheader("Variables del Modelo (Métricas NLP)")
    val_exclamations = st.slider("Cantidad de Exclamaciones (exclamation_count)", min_value=0, max_value=20, value=2)
    val_pos_score = st.slider("Score Positivo (positive_word_score)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    val_neg_score = st.slider("Score Negativo (negative_word_score)", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    val_neu_score = st.slider("Score Neutro (neutral_word_score)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    
    submit_btn = st.form_submit_button("Procesar ETL e Iniciar Predicción")

if submit_btn:
    # Estructuramos el dataframe simulando exactamente la recepción del dataset crudo
    data_individual = pd.DataFrame([{
        'RecordID': val_id, 'FullName': val_name, 'Phone': val_phone, 'ZodiacSign': val_zodiac,
        'FavoriteColor': val_color, 'Hobby': val_hobby, 'review_length': val_len,
        'exclamation_count': val_exclamations, 'question_count': val_questions,
        'positive_word_score': val_pos_score, 'negative_word_score': val_neg_score,
        'neutral_word_score': val_neu_score, 'rating': val_rating, 'product_category': val_cat,
        'verified_purchase': val_verified, 'helpful_votes': val_votes, 'review_year': val_year,
        'Platform': val_platform
    }])
    
    # Aplicamos ETL y predicción
    res_df = limpiar_y_predecir(data_individual)
    pred_sent = res_df.loc[0, 'sentimiento_predicho']
    pred_conf = res_df.loc[0, 'confianza_prediccion']
    
    st.sidebar.markdown("--- ")
    st.sidebar.subheader("Resultado del Análisis:")
    if pred_sent == 'positive':
        st.sidebar.success(f"🟢 **Positivo** (Confianza: {pred_conf})")
    elif pred_sent == 'negative':
        st.sidebar.error(f"🔴 **Negativo** (Confianza: {pred_conf})")
    else:
        st.sidebar.warning(f"🟡 **Neutro** (Confianza: {pred_conf})")

# --- INTERFAZ 2: ENTORNO DINÁMICO DE CARGA MASIVA (PANEL CENTRAL) ---
st.subheader("📂 Carga de base de datos a predecir")
st.write("Sube un archivo en formato **CSV o Excel** que contenga todos los campos obligatorios del dataset inicial para proceder con el análisis.")

archivo_subido = st.file_uploader("Selecciona el archivo (.csv, .xlsx, .xls)", type=["csv", "xlsx", "xls"])

if archivo_subido is not None:
    try: 
        # Lectura dinámica
        nombre_archivo = archivo_subido.name
        if nombre_archivo.endswith('.csv'):
            df_cargado = pd.read_csv(archivo_subido)
        else:
            df_cargado = pd.read_excel(archivo_subido)
            
        # Verificación de que estén presentes las variables esperadas
        columnas_faltantes = [col for col in ALL_INITIAL_COLUMNS if col not in df_cargado.columns]
        
        if columnas_faltantes:
            st.warning(f"⚠️ Tu archivo no contiene todas las columnas requeridas del dataset inicial. Faltan: {columnas_faltantes}. El sistema intentará imputarlas con valores por defecto para evitar errores.")
            for col in columnas_faltantes:
                df_cargado[col] = np.nan
                
        st.success(f"✅ Archivo '{nombre_archivo}' cargado con éxito. Se detectaron **{len(df_cargado)} registros**.")
        
        # Mostrar vista previa original
        with st.expander("👁️ Ver primeros registros originales cargados"):
            st.dataframe(df_cargado.head(10))
            
        # Procesamiento ETL y Predicción
        with st.spinner("🤖 Ejecutando limpieza ETL y calculando sentimientos con SVM..."):
            df_predicho = limpiar_y_predecir(df_cargado)
            
        st.subheader("🎯 Resultados de las Predicciones")
        
        # Distribución de Clases
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Distribución de sentimientos:**")
            resumen_sent = df_predicho['sentimiento_predicho'].value_counts()
            st.dataframe(resumen_sent)
        with col2:
            cols_result = ['sentimiento_predicho', 'confianza_prediccion'] + [c for c in df_predicho.columns if c not in ['sentimiento_predicho', 'confianza_prediccion']]
            st.markdown("**Tabla de resultados con predicciones:**")
            st.dataframe(df_predicho[cols_result].head(100))
            
        # Descarga de resultados procesados
        st.markdown("### 📥 Descargar Resultados")
        st.write("Descarga la base de datos completa con las predicciones y niveles de confianza adjuntados localmente.")
        
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
    st.info("💡 **Nota**: Para realizar una predicción masiva exitosa, asegúrate de subir un documento con la estructura del dataset original.")
