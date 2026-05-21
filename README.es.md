# Clasificador de Adicción al Smartphone

> Un clasificador de regresión logística que predice el riesgo de adicción al smartphone basándose en hábitos de uso diario — desplegado como una aplicación web interactiva con Streamlit.

**Demo en vivo:** [ml-webapp-project-1.onrender.com](https://ml-webapp-project-1.onrender.com/)

---

## Problema

El uso excesivo del smartphone se ha convertido en una preocupación creciente de salud pública, pero la mayoría de las personas no tienen forma de evaluar objetivamente si sus hábitos los ponen en riesgo. Este proyecto construye un clasificador binario que toma métricas simples de uso diario — tiempo de pantalla, aperturas de aplicaciones, horas en redes sociales — y predice si el patrón de comportamiento del usuario es consistente con la adicción al smartphone. El objetivo es hacer esa señal accesible a través de una interfaz limpia e interactiva que cualquiera pueda usar.

## Conjunto de datos

- **Fuente:** [Kaggle — Smartphone Usage & Addiction Analysis](https://www.kaggle.com/) *(actualizar con el enlace directo al dataset)*
- **Tamaño:** 7.500 filas × 15 columnas
- **Variable objetivo:** `addicted_label` (0 = No adicto, 1 = Adicto)
- **Características clave:** `daily_screen_time_hours`, `weekend_screen_time`, `social_media_hours`, `app_opens_per_day`, `gaming_hours`, `sleep_hours`, `notifications_per_day`, `stress_level`, `academic_work_impact`

## Metodología

1. **EDA (10 pasos):** Se exploraron distribuciones, correlaciones y balance de clases. Se detectó que el dataset es sintético — todas las características tienen distribuciones casi uniformes, lo que limita la validez en el mundo real pero permite al modelo aprender patrones significativos.
2. **Análisis de características:** Se construyeron mapas de calor de correlación y diagramas de caja de cada variable frente al objetivo. `daily_screen_time_hours` (r = 0,58) y `weekend_screen_time` (r = 0,56) son los predictores más fuertes. Las variables categóricas (género, nivel de estrés) muestran correlación casi nula.
3. **Ingeniería de características:** Codificación ordinal de `stress_level` y `academic_work_impact`, codificación one-hot de `gender`, y aplicación de `StandardScaler` a todas las variables numéricas.
4. **Selección de características:** Uso de `SelectKBest` con puntuación chi-cuadrado — confirmó que el tiempo de pantalla y las horas en redes sociales impulsan la señal predictiva.
5. **Modelado:** Regresión logística con penalización ElasticNet (`solver=saga`). `GridSearchCV` de 5 pliegues sobre `C`, `l1_ratio` y `class_weight`, optimizando el AUC-ROC.
6. **Despliegue:** Modelo y escalador serializados con `pickle`, e integrados en una app Streamlit que recibe entradas del usuario y devuelve una predicción con probabilidades de confianza.

## Resultados

El modelo final fue seleccionado mediante validación cruzada de 5 pliegues optimizando AUC-ROC. `daily_screen_time_hours` y `weekend_screen_time` tienen los coeficientes positivos más grandes — consistente con el análisis exploratorio. La app Streamlit muestra el desglose de confianza (ej. "No adicto: 98,4% | Adicto: 1,6%") junto a cada predicción, haciendo el resultado transparente en lugar de solo una etiqueta.

## Tecnologías utilizadas

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Streamlit` · `SQLite` · `Matplotlib` · `Seaborn` · `pickle`

## Ejecución local

```bash
git clone https://github.com/matthewkane-ml/ML_WebApp_MTK.git
cd ML_WebApp_MTK
pip install -r requirements.txt

# Ejecutar el EDA y entrenar el modelo primero
python src/ML_WebAPP.py

# Lanzar la app Streamlit
streamlit run src/streamlit_app.py
```

## Capturas de pantalla

![App Streamlit — interfaz de predicción de adicción](screenshots/app_prediction.png)

## Próximos pasos

- Entrenar con datos reales (ej. exportaciones de Screen Time de iOS o Digital Wellbeing de Android) para mejorar la validez más allá de datos sintéticos
- Añadir explicabilidad con SHAP para que los usuarios vean qué hábito específico impulsa su puntuación de riesgo
- Probar modelos basados en árboles (Random Forest, XGBoost) para capturar interacciones no lineales entre características

---

**Autor:** Matthew Kane — [LinkedIn](https://www.linkedin.com/in/thomas-k-392094410/) · [Portafolio GitHub](https://github.com/matthewkane-ml)
