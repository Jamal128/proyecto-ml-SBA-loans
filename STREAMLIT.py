# app.py - Aplicación Completa y Explicativa para Predicción de Monto Aprobado y Estado de Préstamo por SBA

import streamlit as st
import pandas as pd
import numpy as np
import joblib
# Importar las clases exactas usadas en tu pipeline de regresión y clasificación
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer # Añadido FunctionTransformer por si to_string_func lo usa
from sklearn.impute import SimpleImputer
import xgboost as xgb

import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Configurar matplotlib para evitar problemas con Streamlit
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'

# --- INICIO: Definición de la función 'to_string_func' ---
# !! IMPORTANTE !!
# Reemplaza este placeholder con la definición EXACTA de 'to_string_func'
# que usaste cuando guardaste tu modelo 'final_model_pipeline_xgb_cls.pkl'.
# Si esta función no es correcta, el modelo de clasificación seguirá sin cargarse.

def to_string_func(x):
    """Convierte la entrada a tipo string. Usada en FunctionTransformer."""
    return x.astype(str)

# --- FIN: Definición de la función 'to_string_func' ---


# --- Configuración de la Página Streamlit ---
st.set_page_config(page_title="Predicción Préstamos SBA", layout="wide")

# (El resto de tus funciones como create_date_features, load_regression_model, etc. permanecen aquí)
# --- Función para crear características de fecha (Debe ser idéntica a la del notebook) ---
# Adaptada para manejar pd.NaT
def create_date_features(df_input, date_col, approval_date_col='Fecha_Aprobacion'):
    df = df_input.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce') # 'coerce' convierte errores a NaT

        # Extraer componentes de fecha - resultarán en NaN/NaT si la fecha original era NaT
        temp_date_col = df[date_col].fillna(pd.Timestamp('1900-01-01')) # Rellenar temporalmente para usar .dt accessor

        df[f'{date_col}_Year'] = temp_date_col.dt.year
        df[f'{date_col}_Month'] = temp_date_col.dt.month
        df[f'{date_col}_Day'] = temp_date_col.dt.day
        df[f'{date_col}_DayOfWeek'] = temp_date_col.dt.dayofweek

        # Calcular duración si la columna de fecha de aprobación existe y es datetime, y la fecha actual NO es NaT
        ap_col_for_duration = approval_date_col if approval_date_col in df.columns else f'{approval_date_col}_Year'
        if ap_col_for_duration in df.columns and pd.api.types.is_datetime64_any_dtype(df[ap_col_for_duration]):
            df[ap_col_for_duration] = pd.to_datetime(df[ap_col_for_duration], errors='coerce')
            valid_duration_mask = df[date_col].notna() & df[ap_col_for_duration].notna()
            duration = pd.Series(np.nan, index=df.index)
            if valid_duration_mask.any():
                duration[valid_duration_mask] = (df.loc[valid_duration_mask, date_col] - df.loc[valid_duration_mask, ap_col_for_duration]).dt.days
            df[f'{date_col}_Duration_Days_From_Approval'] = duration.clip(lower=0)

        df = df.drop(columns=[date_col])

    if date_col == approval_date_col and approval_date_col in df.columns:
        df = df.drop(columns=[approval_date_col])

    return df


# --- Cargar el modelo de regresión optimizado ---
@st.cache_resource # Cargar el modelo una sola vez
def load_regression_model(model_path='best_model_regresion_xgb.pkl'): # Asegúrate que este es el nombre correcto
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"¡Error al cargar el archivo del modelo de regresión! Asegúrate de que '{model_path}' esté en el directorio correcto.")
        st.stop()
    except Exception as e:
        st.error(f"Error inesperado al cargar el modelo de regresión: {e}")
        st.stop()

model_regresion = load_regression_model()
st.sidebar.success("Modelo de predicción de monto cargado.")

try:
    preprocessor_trained_reg = model_regresion.named_steps['preprocessor']
    regressor_model_xgb = model_regresion.named_steps['regressor']
except KeyError:
    st.error("El pipeline de regresión no tiene los pasos 'preprocessor' o 'regressor' con esos nombres. Verifica la estructura de tu pipeline.")
    st.stop()
except AttributeError:
    st.error("El objeto cargado como 'model_regresion' no parece ser un Pipeline de scikit-learn.")
    st.stop()


# --- Cargar el modelo de CLASIFICACIÓN optimizado ---
@st.cache_resource
def load_classification_model(model_path='final_model_pipeline_xgb_cls.pkl'):
    try:
        # La función to_string_func debe estar definida globalmente ANTES de esta llamada
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"¡Error al cargar el archivo del modelo de clasificación! Asegúrate de que '{model_path}' esté en el directorio correcto.")
        st.stop()
    except AttributeError as e:
        st.error(f"Error de atributo al cargar el modelo de clasificación: {e}")
        st.error("Esto usualmente significa que una función o clase personalizada (como 'to_string_func') usada en el modelo no está definida en este script. Por favor, verifica la definición de 'to_string_func' al inicio del script.")
        st.stop()
    except Exception as e:
        st.error(f"Error inesperado al cargar el modelo de clasificación: {e}")
        st.stop()

model_clasificacion = load_classification_model()
st.sidebar.success("Modelo de predicción de aprobación cargado.")


# --- Cargar una muestra de los datos originales para Contexto y Gráficos ---
@st.cache_data
def load_sample_data(sample_path='tu_muestra_datos_originales.parquet'):
    try:
        df_sample = pd.read_parquet(sample_path)
        return df_sample
    except FileNotFoundError:
        st.warning(f"Archivo de muestra ('{sample_path}') no encontrado. Los gráficos contextuales no estarán disponibles.")
        return None
    except Exception as e:
        st.warning(f"Error al cargar archivo de muestra '{sample_path}': {e}. Gráficos contextuales no disponibles.")
        return None

df_sample_original = load_sample_data('tu_muestra_datos_originales.parquet')


# --- Cargar el Explicador SHAP ---
@st.cache_resource
def load_shap_explainer(_xgb_model=None): # Argumento con guion bajo
    try:
        if _xgb_model is None:
            st.error("No se proporcionó el modelo XGBoost base para el explicador SHAP.")
            return None
        explainer = shap.TreeExplainer(_xgb_model)
        return explainer
    except Exception as e:
        st.error(f"Error al crear el explicador SHAP: {e}")
        return None


# --- Título de la Aplicación ---
st.title("📈 Predicción y Análisis de Préstamos SBA")
st.write("""
Esta aplicación te permite estimar:
1.  Si una solicitud de préstamo probablemente será **ACEPTADA o RECHAZADA** (predicción principal).
2.  El **monto** que la Small Business Administration (SBA) podría aprobar.

Introduce los detalles de la solicitud en la barra lateral y haz clic en "Realizar Predicción".
Además de la predicción, podrás explorar qué factores influyen más en la estimación del monto.
""")

st.sidebar.header("📝 Datos de la Solicitud de Préstamo")
st.sidebar.info("Introduce la información disponible al momento de solicitar el préstamo.")

# --- Interfaz de Usuario para Entrada de Datos ---
user_input = {}
st.sidebar.write("#### 🏢 Información de la Empresa:")
user_input['Estado_Empresa'] = st.sidebar.text_input("Estado de la Empresa (ej. CA, NY)", "CA", key='estado_empresa_input', max_chars=2).upper()
user_input['Codigo_Postal_Empresa'] = st.sidebar.number_input("Código Postal de la Empresa", value=90210, step=1, min_value=0, max_value=99999, key='zip_empresa_input')
user_input['Codigo_NAICS'] = st.sidebar.number_input("Código NAICS (Primeros 2 dígitos)", value=54, step=1, min_value=11, max_value=99, help="Los dos primeros dígitos del código de la industria (ej. 54 para Servicios Profesionales)", key='naics_input')

st.sidebar.write("#### 🏦 Información del Banco:")
user_input['Estado_Banco'] = st.sidebar.text_input("Estado del Banco (ej. CA, NY)", "CA", key='estado_banco_input', max_chars=2).upper()

st.sidebar.write("#### 📄 Detalles de la Solicitud:")
user_input['Fecha_Aprobacion'] = st.sidebar.date_input("Fecha de Aprobación Estimada/Real", value=datetime.date.today(), help="Fecha en que se espera o fue aprobado el préstamo por la SBA", key='fecha_aprobacion_input')
user_input['Plazo_Meses'] = st.sidebar.number_input("Plazo en Meses", value=90, min_value=1, max_value=300, help="Duración del préstamo en meses", key='plazo_meses_input')
user_input['Numero_Empleados'] = st.sidebar.number_input("Número de Empleados al Solicitar", value=5, min_value=0, max_value=10000, help="Número de empleados a tiempo completo", key='num_empleados_input')
user_input['Negocio_Existente'] = st.sidebar.selectbox("Tipo de Negocio", options=[1, 2], format_func=lambda x: "Existente" if x == 1 else "Nuevo", help="1 para negocio existente, 2 para nuevo negocio", key='negocio_existente_input')
user_input['Empleos_Creados'] = st.sidebar.number_input("Empleos a Crear", value=5, min_value=0, max_value=1000, help="Número de empleos que se espera crear", key='empleos_creados_input')
user_input['Empleos_Retenidos'] = st.sidebar.number_input("Empleos a Retener", value=1, min_value=0, max_value=5000, help="Número de empleos que se espera retener", key='empleos_retenidos_input')
user_input['Codigo_Franquicia'] = st.sidebar.number_input("Código de Franquicia", value=1, min_value=0, max_value=99999, help="Código de franquicia (1 si no es franquicia/desconocido)", key='franquicia_input')
user_input['Area_UrbanRural'] = st.sidebar.selectbox("Área de Ubicación", options=[1, 2, 0], format_func=lambda x: "Urbana" if x==1 else ("Rural" if x==2 else "Indefinido"), help="1=Urbana, 2=Rural, 0=Indefinido", key='area_input')
user_input['Linea_Credito_Rotativa'] = st.sidebar.selectbox("Solicita Línea de Crédito Rotativa", options=["Y", "N"], index=1, help="Y si es línea de crédito rotativa, N si no", key='revlinecr_input')
user_input['Programa_LowDoc'] = st.sidebar.selectbox("Programa LowDoc", options=["Y", "N"], index=1, help="Y si fue bajo programa LowDoc, N si no", key='lowdoc_input')

st.sidebar.write("#### 💰 Montos Solicitados/Propuestos:")
user_input['Monto_Aprobado_Banco'] = st.sidebar.number_input("Monto Aprobado por el Banco (Propuesto)", value=500000.0, min_value=0.0, step=1000.0, help="Monto del préstamo aprobado/propuesto por el banco", key='monto_aprobado_banco_input')

user_input['Fecha_Desembolso'] = pd.NaT
user_input['Monto_Desembolsado'] = 0.0

predict_button = st.sidebar.button("🚀 Realizar Predicción", key='predict_button', type="primary")

# --- Lógica para procesar la entrada y realizar la predicción ---
if predict_button:
    st.subheader("🎯 Resultados de la Predicción")

    input_df_raw = pd.DataFrame([user_input])
    numeric_cols_raw = ['Codigo_Postal_Empresa', 'Codigo_NAICS', 'Plazo_Meses', 'Numero_Empleados',
                        'Empleos_Creados', 'Empleos_Retenidos', 'Codigo_Franquicia',
                        'Monto_Aprobado_Banco', 'Monto_Desembolsado']
    datetime_cols_raw = ['Fecha_Aprobacion', 'Fecha_Desembolso']

    for col in numeric_cols_raw:
        if col in input_df_raw.columns: input_df_raw[col] = pd.to_numeric(input_df_raw[col], errors='coerce')
    for col in datetime_cols_raw:
        if col in input_df_raw.columns: input_df_raw[col] = pd.to_datetime(input_df_raw[col], errors='coerce')

    processed_features_dict = {}
    original_cols_direct = ['Codigo_Postal_Empresa', 'Codigo_NAICS', 'Plazo_Meses', 'Numero_Empleados',
                            'Empleos_Creados', 'Empleos_Retenidos', 'Codigo_Franquicia',
                            'Monto_Desembolsado', 'Monto_Aprobado_Banco', 'Estado_Empresa', 'Estado_Banco',
                            'Negocio_Existente', 'Area_UrbanRural', 'Linea_Credito_Rotativa', 'Programa_LowDoc']
    for col in original_cols_direct:
        if col in input_df_raw.columns: processed_features_dict[col] = input_df_raw[col].iloc[0]
        else: st.error(f"Error interno: Columna '{col}' no encontrada."); st.stop()

    ap_date = input_df_raw.get('Fecha_Aprobacion', pd.NaT).iloc[0]
    dis_date = input_df_raw.get('Fecha_Desembolso', pd.NaT).iloc[0]

    if pd.isna(ap_date):
        processed_features_dict.update({
            'Fecha_Aprobacion_Year': np.nan, 'Fecha_Aprobacion_Month': np.nan,
            'Fecha_Aprobacion_Day': np.nan, 'Fecha_Aprobacion_DayOfWeek': np.nan,
            'Fecha_Aprobacion_Missing': 1, 'ApprovalYear': np.nan, 'Anho_Fiscal_Aprobacion': 'Unknown'
        })
    else:
        processed_features_dict.update({
            'Fecha_Aprobacion_Year': ap_date.year, 'Fecha_Aprobacion_Month': ap_date.month,
            'Fecha_Aprobacion_Day': ap_date.day, 'Fecha_Aprobacion_DayOfWeek': ap_date.dayofweek,
            'Fecha_Aprobacion_Missing': 0, 'ApprovalYear': ap_date.year,
            'Anho_Fiscal_Aprobacion': str(ap_date.year)
        })

    if pd.isna(dis_date):
        processed_features_dict.update({
            'Fecha_Desembolso_Year': np.nan, 'Fecha_Desembolso_Month': np.nan,
            'Fecha_Desembolso_Day': np.nan, 'Fecha_Desembolso_DayOfWeek': np.nan,
            'Fecha_Desembolso_Missing': 1, 'Fecha_Desembolso_Duration_Days_From_Approval': np.nan
        })
    else:
        duration_val = (dis_date - ap_date).days if pd.notna(ap_date) else np.nan
        processed_features_dict.update({
            'Fecha_Desembolso_Year': dis_date.year, 'Fecha_Desembolso_Month': dis_date.month,
            'Fecha_Desembolso_Day': dis_date.day, 'Fecha_Desembolso_DayOfWeek': dis_date.dayofweek,
            'Fecha_Desembolso_Missing': 0,
            'Fecha_Desembolso_Duration_Days_From_Approval': max(0, duration_val) if pd.notna(duration_val) else np.nan
        })

    expected_pipeline_cols = [
        'Codigo_Postal_Empresa', 'Codigo_NAICS', 'Plazo_Meses', 'Numero_Empleados',
        'Empleos_Creados', 'Empleos_Retenidos', 'Codigo_Franquicia',
        'Monto_Desembolsado', 'Monto_Aprobado_Banco',
        'Estado_Empresa', 'Estado_Banco', 'Negocio_Existente',
        'Area_UrbanRural', 'Linea_Credito_Rotativa', 'Programa_LowDoc',
        'Anho_Fiscal_Aprobacion', 'ApprovalYear',
        'Fecha_Aprobacion_Year', 'Fecha_Aprobacion_Month', 'Fecha_Aprobacion_Day', 'Fecha_Aprobacion_DayOfWeek', 'Fecha_Aprobacion_Missing',
        'Fecha_Desembolso_Year', 'Fecha_Desembolso_Month', 'Fecha_Desembolso_Day', 'Fecha_Desembolso_DayOfWeek', 'Fecha_Desembolso_Duration_Days_From_Approval', 'Fecha_Desembolso_Missing'
    ]
    final_input_data = {col: processed_features_dict.get(col, np.nan) for col in expected_pipeline_cols}
    processed_input_for_pipeline = pd.DataFrame([final_input_data], columns=expected_pipeline_cols)

    # --- Predicción de Clasificación (Estado del Préstamo) ---
    prediction_successful_cls = False
    try:
        pred_clasificacion_proba = model_clasificacion.predict_proba(processed_input_for_pipeline)
        pred_clasificacion_label_num = model_clasificacion.predict(processed_input_for_pipeline)

        prob_aprobado = pred_clasificacion_proba[0][1] * 100
        estado_predicho_texto = "✔️ Aprobado Probablemente" if pred_clasificacion_label_num[0] == 1 else "❌ Rechazado Probablemente"
        
        st.metric(
            label="Estado del Préstamo Estimado (Principal)",
            value=estado_predicho_texto,
            help=f"Probabilidad de aprobación: {prob_aprobado:.2f}%"
        )
        if pred_clasificacion_label_num[0] == 1 :
             st.success(f"El préstamo tiene una probabilidad de aprobación del {prob_aprobado:.2f}%.")
        else:
            st.error(f"El préstamo tiene una probabilidad de rechazo del {100-prob_aprobado:.2f}%. (Probabilidad de aprobación: {prob_aprobado:.2f}%)")
        prediction_successful_cls = True

    except Exception as e:
        st.error(f"Error al realizar la predicción de clasificación: {e}")
        from io import StringIO # Para mostrar info del DataFrame
        buffer_cls = StringIO()
        processed_input_for_pipeline.info(buf=buffer_cls)
        st.text(f"Info del DataFrame de entrada al pipeline de CLASIFICACIÓN:\n{buffer_cls.getvalue()}")
        st.write("DataFrame de entrada (primeras filas):", processed_input_for_pipeline.head())


    st.markdown("---")

    # --- Predicción de Regresión (Monto Aprobado) ---
    prediction_successful_reg = False
    if prediction_successful_cls: # Solo intentar si la clasificación fue bien, o manejar independientemente
        try:
            input_data_transformed_reg = preprocessor_trained_reg.transform(processed_input_for_pipeline)
            pred_regresion = regressor_model_xgb.predict(input_data_transformed_reg)

            resultado_col1, resultado_col2 = st.columns([1, 1])
            with resultado_col1:
                st.metric(label="💰 Monto Aprobado por SBA Estimado", value=f"${pred_regresion[0]:,.2f}")
            with resultado_col2:
                fig, ax = plt.subplots(figsize=(5, 0.8))
                base_display_max = df_sample_original['Monto_Aprobado_SBA'].quantile(0.95) if df_sample_original is not None and 'Monto_Aprobado_SBA' in df_sample_original else 250000
                max_value = max(base_display_max, pred_regresion[0] * 1.5, 50000)
                ax.barh(0, max_value, color='lightgray', height=0.5, edgecolor='gray')
                ax.barh(0, pred_regresion[0], color='#1f77b4', height=0.5, edgecolor='black')
                ax.set_xlim(0, max_value)
                ax.set_yticks([])
                num_ticks = 5
                tick_values = np.linspace(0, max_value, num_ticks)
                tick_labels = [f'${val:,.0f}' for val in tick_values]
                ax.set_xticks(tick_values)
                ax.set_xticklabels(tick_labels, fontsize=8)
                plt.title("Visualización del Monto Estimado", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            prediction_successful_reg = True
        except Exception as e:
            st.error(f"Error al realizar la predicción de regresión: {e}")
            from io import StringIO
            buffer_reg = StringIO()
            processed_input_for_pipeline.info(buf=buffer_reg)
            st.text(f"Info del DataFrame de entrada al pipeline de REGRESIÓN (antes de preprocesador interno):\n{buffer_reg.getvalue()}")
            st.write("DataFrame de entrada (primeras filas):", processed_input_for_pipeline.head())
    else:
        st.warning("No se intentó la predicción de monto debido a un error en la predicción de estado de aprobación.")


    st.markdown("---")

    # --- SHAP: Explicación de la predicción de MONTO ---
    # --- SHAP: Explicación de la predicción de MONTO ---
    if prediction_successful_reg: # Solo mostrar SHAP si la predicción de regresión fue exitosa
        st.subheader("🔍 Explicación de la Predicción del Monto (SHAP)")
        st.write("Descubre cómo cada característica influyó en la estimación del *monto aprobado*.")

        try:
            shap_explainer_reg = load_shap_explainer(_xgb_model=regressor_model_xgb)
            if shap_explainer_reg is not None:
                if hasattr(preprocessor_trained_reg, 'get_feature_names_out'):
                    feature_names_out_reg = preprocessor_trained_reg.get_feature_names_out()
                else: # Fallback
                    try:
                        feature_names_out_reg = []
                        for name, trans, cols in preprocessor_trained_reg.transformers_:
                            if trans == 'drop' or trans == 'passthrough': continue
                            if hasattr(trans, 'get_feature_names_out'): feature_names_out_reg.extend(trans.get_feature_names_out(cols))
                            else: feature_names_out_reg.extend(cols)
                        if not feature_names_out_reg : feature_names_out_reg = [f'feature_{i}' for i in range(input_data_transformed_reg.shape[1])]
                    except Exception: feature_names_out_reg = [f'feature_{i}' for i in range(input_data_transformed_reg.shape[1])]

                # input_data_transformed_reg es la salida 2D del preprocesador
                if hasattr(input_data_transformed_reg, 'toarray'):
                    input_data_dense_2d_reg = input_data_transformed_reg.toarray() # Asegura que sea un array denso 2D
                else:
                    input_data_dense_2d_reg = input_data_transformed_reg # Ya debería ser un array numpy 2D

                # Verificar que input_data_dense_2d_reg sea 2D
                if input_data_dense_2d_reg.ndim != 2 or input_data_dense_2d_reg.shape[0] != 1:
                    st.error(f"Error interno: input_data_dense_2d_reg no tiene la forma esperada (1, N). Su forma es: {input_data_dense_2d_reg.shape}")
                    st.stop()

                if len(feature_names_out_reg) != input_data_dense_2d_reg.shape[1]:
                    st.error(f"Discrepancia en nombres de características ({len(feature_names_out_reg)}) y datos transformados ({input_data_dense_2d_reg.shape[1]}) para SHAP regresión.")
                else:
                    # --- INICIO DE LA CORRECCIÓN ---
                    # Calcular valores SHAP pasando el array 2D (1, num_features)
                    shap_values_output_2d = shap_explainer_reg.shap_values(input_data_dense_2d_reg)
                    # Para regresión de una sola salida, shap_values_output_2d tendrá forma (1, num_features)
                    # Para la Explanation de una sola instancia, necesitamos la primera (y única) fila como un array 1D.
                    shap_values_for_instance_1d = shap_values_output_2d[0]
                    # Los datos de la característica para la instancia también deben ser 1D para shap.Explanation
                    data_for_instance_1d = input_data_dense_2d_reg[0,:]
                    # --- FIN DE LA CORRECCIÓN ---

                    fig_waterfall, ax_waterfall = plt.subplots()
                    shap.waterfall_plot(shap.Explanation(values=shap_values_for_instance_1d, # Array 1D de valores SHAP
                                                         base_values=shap_explainer_reg.expected_value, # Escalar
                                                         data=data_for_instance_1d, # Array 1D de datos de la instancia
                                                         feature_names=feature_names_out_reg),
                                        max_display=15, show=False)
                    plt.tight_layout()
                    st.pyplot(fig_waterfall)
                    plt.close(fig_waterfall)

                    st.write("##### Detalles de la Influencia de Características (Monto):")
                    # Usar los valores SHAP 1D para el DataFrame
                    shap_values_df_reg = pd.DataFrame({'feature': feature_names_out_reg,
                                                       'shap_value': shap_values_for_instance_1d, # Usar el array 1D
                                                       'abs_shap_value': np.abs(shap_values_for_instance_1d)} # Usar el array 1D
                                                      ).sort_values(by='abs_shap_value', ascending=False).head(15)
                    col1_shap, col2_shap = st.columns(2)
                    with col1_shap:
                        st.write("**📈 Aumentaron el monto:**")
                        pos_impact_reg = shap_values_df_reg[shap_values_df_reg['shap_value'] > 1e-6]
                        if not pos_impact_reg.empty:
                            for _, row in pos_impact_reg.iterrows():
                                orig_feat_name = row['feature'].split('__')[-1]
                                orig_val = processed_input_for_pipeline.iloc[0].get(orig_feat_name, "N/A")
                                st.markdown(f"- **{row['feature'].replace('_', ' ').title()}**: `{orig_val}` (+${row['shap_value']:,.2f})")
                        else: st.write("Ninguna característica principal aumentó significativamente el monto.")
                    with col2_shap:
                        st.write("**📉 Disminuyeron el monto:**")
                        neg_impact_reg = shap_values_df_reg[shap_values_df_reg['shap_value'] < -1e-6]
                        if not neg_impact_reg.empty:
                            for _, row in neg_impact_reg.iterrows():
                                orig_feat_name = row['feature'].split('__')[-1]
                                orig_val = processed_input_for_pipeline.iloc[0].get(orig_feat_name, "N/A")
                                st.markdown(f"- **{row['feature'].replace('_', ' ').title()}**: `{orig_val}` (${row['shap_value']:,.2f})")
                        else: st.write("Ninguna característica principal disminuyó el monto.")
                    st.caption("Los nombres pueden incluir prefijos del preprocesamiento.")
            else:
                st.warning("Explicador SHAP para regresión no cargado. Explicaciones no disponibles.")
        except Exception as e_shap:
            st.error(f"Error al generar explicación SHAP para regresión: {e_shap}")
            if 'input_data_dense_2d_reg' in locals():
                 st.write(f"Debug SHAP: Forma de input_data_dense_2d_reg: {input_data_dense_2d_reg.shape}") # Para depurar forma
        st.markdown("---")


# --- Sección de Análisis Exploratorio de Datos (si hay datos de muestra) ---
if df_sample_original is not None:
    st.header("📊 Análisis Exploratorio de Datos de Muestra")
    st.write("Visualizaciones basadas en una muestra de datos históricos para dar contexto.")
    col_eda1, col_eda2 = st.columns(2)

    with col_eda1:
        st.write("#### Distribución del Monto Aprobado por SBA (Muestra)")
        if 'Monto_Aprobado_SBA' in df_sample_original:
            fig_dist_sba, ax_dist_sba = plt.subplots()
            monto_sba_sample = df_sample_original['Monto_Aprobado_SBA'][df_sample_original['Monto_Aprobado_SBA'] < df_sample_original['Monto_Aprobado_SBA'].quantile(0.99)]
            sns.histplot(monto_sba_sample, kde=True, ax=ax_dist_sba, bins=50)
            ax_dist_sba.set_title("Distribución de Montos Aprobados por SBA")
            ax_dist_sba.set_xlabel("Monto Aprobado SBA ($)")
            ax_dist_sba.set_ylabel("Frecuencia")
            st.pyplot(fig_dist_sba)
            plt.close(fig_dist_sba)
        else:
            st.warning("Columna 'Monto_Aprobado_SBA' no encontrada en los datos de muestra.")

    with col_eda2:
        st.write("#### Monto Aprobado por Banco vs. SBA (Muestra)")
        if 'Monto_Aprobado_Banco' in df_sample_original and 'Monto_Aprobado_SBA' in df_sample_original:
            fig_scatter_bank_sba, ax_scatter_bank_sba = plt.subplots()
            sample_subset = df_sample_original[
                (df_sample_original['Monto_Aprobado_Banco'] < df_sample_original['Monto_Aprobado_Banco'].quantile(0.99)) &
                (df_sample_original['Monto_Aprobado_SBA'] < df_sample_original['Monto_Aprobado_SBA'].quantile(0.99))
            ].sample(min(1000, len(df_sample_original)))

            sns.scatterplot(data=sample_subset, x='Monto_Aprobado_Banco', y='Monto_Aprobado_SBA', alpha=0.5, ax=ax_scatter_bank_sba)
            ax_scatter_bank_sba.set_title("Monto Banco vs. Monto SBA")
            ax_scatter_bank_sba.set_xlabel("Monto Aprobado por Banco ($)")
            ax_scatter_bank_sba.set_ylabel("Monto Aprobado por SBA ($)")
            max_val_scatter = sample_subset[['Monto_Aprobado_Banco', 'Monto_Aprobado_SBA']].max().max()
            ax_scatter_bank_sba.plot([0, max_val_scatter], [0, max_val_scatter], 'r--', lw=2, label='Línea de Igualdad')
            ax_scatter_bank_sba.legend()
            st.pyplot(fig_scatter_bank_sba)
            plt.close(fig_scatter_bank_sba)
        else:
            st.warning("Columnas 'Monto_Aprobado_Banco' o 'Monto_Aprobado_SBA' no encontradas.")

    col_eda3, col_eda4 = st.columns(2)
    with col_eda3:
        st.write("#### Monto Aprobado SBA por Tipo de Negocio (Muestra)")
        if 'Negocio_Existente' in df_sample_original and 'Monto_Aprobado_SBA' in df_sample_original:
            fig_box_tipo, ax_box_tipo = plt.subplots()
            df_sample_original['Tipo_Negocio_Label'] = df_sample_original['Negocio_Existente'].map({1: 'Existente', 2: 'Nuevo'}).fillna('Desconocido')
            sns.boxplot(data=df_sample_original[df_sample_original['Monto_Aprobado_SBA'] < df_sample_original['Monto_Aprobado_SBA'].quantile(0.99)],
                        x='Tipo_Negocio_Label', y='Monto_Aprobado_SBA', ax=ax_box_tipo)
            ax_box_tipo.set_title("Monto SBA por Tipo de Negocio")
            ax_box_tipo.set_xlabel("Tipo de Negocio")
            ax_box_tipo.set_ylabel("Monto Aprobado SBA ($)")
            st.pyplot(fig_box_tipo)
            plt.close(fig_box_tipo)
        else:
            st.warning("Columnas 'Negocio_Existente' o 'Monto_Aprobado_SBA' no encontradas.")

    with col_eda4:
        st.write("#### Monto Aprobado SBA por Plazo (Muestra)")
        if 'Plazo_Meses' in df_sample_original and 'Monto_Aprobado_SBA' in df_sample_original:
            fig_scatter_plazo, ax_scatter_plazo = plt.subplots()
            sample_subset_plazo = df_sample_original[
                df_sample_original['Monto_Aprobado_SBA'] < df_sample_original['Monto_Aprobado_SBA'].quantile(0.99)
            ].sample(min(1000, len(df_sample_original)))

            sns.scatterplot(data=sample_subset_plazo, x='Plazo_Meses', y='Monto_Aprobado_SBA', alpha=0.5, ax=ax_scatter_plazo)
            ax_scatter_plazo.set_title("Monto SBA vs. Plazo del Préstamo")
            ax_scatter_plazo.set_xlabel("Plazo en Meses")
            ax_scatter_plazo.set_ylabel("Monto Aprobado SBA ($)")
            st.pyplot(fig_scatter_plazo)
            plt.close(fig_scatter_plazo)
        else:
            st.warning("Columnas 'Plazo_Meses' o 'Monto_Aprobado_SBA' no encontradas.")
    st.markdown("---")


# --- Información Adicional y Descargos de Responsabilidad ---
st.header("ℹ️ Información Adicional")
st.info("""
- **Precisión del Modelo:** Estas predicciones se basan en modelos de aprendizaje automático entrenados con datos históricos. No garantizan la aprobación real ni el monto exacto.
- **Factores No Incluidos:** Los modelos pueden no considerar todos los factores que la SBA utiliza en su proceso de decisión final (ej. historial crediticio detallado, plan de negocios completo, condiciones económicas actuales específicas).
- **Uso:** Esta herramienta es para fines informativos y exploratorios. Consulte directamente con la SBA o un asesor financiero para obtener información oficial y asesoramiento.
- **Actualización de Datos:** La precisión de los modelos puede variar con el tiempo.
""")
st.markdown("---")
st.markdown("Desarrollado con Streamlit, XGBoost y SHAP.")
st.markdown("Asegúrate de que las rutas a los archivos de modelo y al archivo de muestra sean correctas.")