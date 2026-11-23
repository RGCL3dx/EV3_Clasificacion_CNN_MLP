# Pipeline Integrado de Clasificación de Razas de Perros (Oxford-IIIT Pet)

Documentación técnica del notebook **Actividad 3.4 - Pipeline Integrado**, diseñado para unificar y estandarizar el flujo de trabajo de visión por computador sobre el dataset **Oxford-IIIT Pet**, comparando arquitecturas **MLP** y **CNN** bajo condiciones controladas, con evaluación de curvas de aprendizaje, auditoría de errores, explicabilidad (**XAI**) y generación de reportes.

---

## 1. Descripción General y Contexto

En escenarios de desarrollo de modelos de *Deep Learning* donde distintos equipos o experimentos han generado variantes de **MLP** y **CNN** con código disperso, la reproducibilidad y la comparación justa suelen verse afectadas. Este pipeline resuelve ese problema mediante:

- **Modularidad por responsabilidad única:** cada etapa del flujo (ingesta, análisis exploratorio, carga y partición, preprocesamiento, construcción del modelo, entrenamiento, evaluación, XAI y reporte) está encapsulada en funciones dedicadas.
- **Reproducibilidad controlada:** fijación de semilla global (`SEED = 36`) y división estratificada *train/test* (`test_size = 0.20`, `random_state = 36`).
- **Trazabilidad y auditoría:** cálculo de métricas múltiples en *test*, diagnóstico automático de curvas de entrenamiento, análisis de confusiones, explicabilidad con **LIME** (MLP) y **Grad-CAM** (CNN), y resumen consolidado mediante `generar_reporte()`.

---

## 2. Dataset y Preprocesamiento

### Fuente de datos
- Dataset **Oxford-IIIT Pet** descargado mediante `kagglehub` (con asociación de *trimaps* y análisis exploratorio inicial sobre el conjunto completo de imágenes, especies y razas).

### Filtrado y partición (`cargar_datos`)
- Selección de la clase/especie **Dog** (`25` razas caninas).
- Redimensionamiento de cada imagen a resolución `(164, 164, 3)` en formato `uint8`.
- Partición estratificada **80% entrenamiento / 20% prueba** (`test_size=0.20`, `stratify=y`, `random_state=36`).

### Preprocesamiento (`preprocesar`)
- Conversión de imágenes a `float32` y escalado al rango `[0.0, 1.0]`.
- Codificación de etiquetas de clase con `LabelEncoder` ajustado únicamente sobre los datos de entrenamiento (`fit_transform` en *train* y `transform` en *test* para evitar fuga de datos).
- Incorporación de bloque de **Data Augmentation** cuando aplica la variante regularizada (`RandomFlip("horizontal")`, `RandomRotation(0.15, fill_mode="reflect")`, `RandomZoom((-0.10, 0.10), fill_mode="reflect")` y `RandomContrast(0.1)`).

---

## 3. Arquitecturas y Estrategias de Regularización

El pipeline permite evaluar tres configuraciones principales (la variante MLP con regularización está explícitamente restringida/no soportada en la función de construcción):

1. **MLP Base (`tipo_modelo="mlp"`, `regularizacion=False`):**
   - Capa `Flatten` sobre la entrada `(164, 164, 3)`.
   - Capas ocultas densas de `256` y `128` unidades con activación `ReLU`.
   - Capa de salida `Dense(num_clases, activation="softmax")`.
2. **CNN Base (`tipo_modelo="cnn"`, `regularizacion=False`):**
   - Cuatro bloques convolucionales con filtros `3x3`, `padding="same"` y activación `ReLU` (`32 -> 64 -> 128 -> 256` filtros).
   - Reducción espacial con `MaxPooling2D((2, 2))` en los tres primeros bloques y `AveragePooling2D((2, 2))` en el cuarto.
   - Capa `Flatten`, capa densa `Dense(256, activation="relu")` y capa de salida `Dense(num_clases, activation="softmax")`.
3. **CNN Regularizada (`tipo_modelo="cnn"`, `regularizacion=True`):**
   - Misma base convolucional enriquecida con **Data Augmentation** y **Dropout** progresivo (`0.3` tras el tercer bloque convolucional, `0.3` tras el cuarto bloque convolucional y `0.4` antes de la capa de salida).

### Callbacks de entrenamiento (`entrenar`)
- **`EarlyStopping`:** monitorea `val_loss`, con `patience=7`, `min_delta=0.001` y `restore_best_weights=True`.
- **`ReduceLROnPlateau`:** monitorea `val_loss`, reduce la tasa de aprendizaje con `factor=0.3`, `patience=3` y `min_lr=1e-7`.

---

## 4. Hiperparámetros y Compilación

| Parámetro | Valor / Configuración | Notas |
| :--- | :--- | :--- |
| Optimizador | `Adam(learning_rate=1e-4)` | Tasa de aprendizaje inicial homogénea |
| Función de pérdida | `sparse_categorical_crossentropy` | Compatible con etiquetas enteras |
| Métrica de compilación | `accuracy` | Monitoreada en entrenamiento y validación |
| Tamaño de lote (`batch_size`) | `64` por defecto en `main()` (`32` por defecto en `entrenar()`) | Configurable vía argumento |
| Épocas (`epochs`) | `20` por defecto en `main()` (`30` por defecto en `entrenar()`) | Acotadas por `EarlyStopping` |
| Semilla aleatoria | `36` | Reproducibilidad de partición y flujo |

---

## 5. Evaluación, Auditoría y Explicabilidad (XAI)

- **Métricas en conjunto de prueba (`evaluar_metricas`):**
  - `Accuracy`, `Top-1`, `Top-5`, `Precision macro`, `Recall macro`, `F1 macro` y `ROC AUC` (`OvR macro`).
  - Matriz de confusión y listado de las **top 15 confusiones** entre razas.
- **Diagnóstico de curvas de aprendizaje (`evaluar_aprendizaje`):**
  - Inspección de evolución de `loss`/`val_loss` y `accuracy`/`val_accuracy`.
  - Detección automática de señales de sobreajuste (por ejemplo, brecha *train-val* superior a `0.30` o incremento sostenido de `val_loss` tras el mínimo) o subajuste (`accuracy` de entrenamiento bajo, p. ej. `< 0.50`).
- **Explicabilidad (`explicar_modelo`):**
  - **LIME** para modelos **MLP**, resaltando superpíxeles con contribución positiva a la predicción.
  - **Grad-CAM** sobre la última capa `Conv2D` para modelos **CNN**, incluyendo el cálculo del porcentaje de activación concentrado en la región central de la imagen (`concentracion_centro`).
- **Reporte ejecutivo (`generar_reporte`):**
  - Consolida métricas, diagnóstico de entrenamiento y criterios de selección de modelo.

---

## 6. Estructura del Flujo y Uso

Las funciones principales del notebook se articulan en torno a `main`:

```python
resultado_mlp = main(tipo_modelo="mlp", regularizacion=False, epochs=20, batch_size=64)
resultado_cnn_base = main(tipo_modelo="cnn", regularizacion=False, epochs=20, batch_size=64)
resultado_cnn_reg = main(tipo_modelo="cnn", regularizacion=True, epochs=20, batch_size=64)
```

### Requisitos habituales
- Python 3.x con `numpy`, `tensorflow` / `keras`, `scikit-learn`, `matplotlib`, `seaborn`, `kagglehub`, `lime` y `opencv-python` / `pillow` (según el entorno de ejecución del notebook).

### Recomendaciones de uso
1. Ejecutar las celdas en orden secuencial para garantizar la descarga del dataset, la creación de variables globales y la consistencia de la semilla.
2. Registrar las salidas de cada corrida en la tabla comparativa de resultados y complementar la auditoría **Grad-CAM** evaluando múltiples muestras (más allá de un índice individual) antes de promover un modelo.