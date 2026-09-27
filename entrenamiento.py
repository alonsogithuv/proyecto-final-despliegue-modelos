from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score, 
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
N_FILAS = 1000
ARCHIVO_DATOS = "datos_manufactura.csv"
ARCHIVO_MODELO = "modelo_produccion.joblib"
UMBRAL = 0.50

df = pd.read_csv(ARCHIVO_DATOS)


# ============================================================
# 2. VARIABLES PREDICTORAS Y TARGET
# ============================================================

TARGET = "defecto_detectado"

FEATURE_COLUMNS = [
    "temperatura",
    "nivel_vibracion",
    "presion_linea",
    "velocidad_produccion",
    "turno_operativo",
    "tipo_maquina",
]

NUMERIC_COLUMNS = [
    "temperatura",
    "nivel_vibracion",
    "presion_linea",
    "velocidad_produccion",
]

CATEGORICAL_COLUMNS = [
    "turno_operativo",
    "tipo_maquina",
]

X = df[FEATURE_COLUMNS].copy()
y = df[TARGET].copy()


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)


# ============================================================
# 4. PREPROCESAMIENTO
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            NUMERIC_COLUMNS,
        ),
        (
            "categorical",
            categorical_transformer,
            CATEGORICAL_COLUMNS,
        ),
    ],
    remainder="drop",
)


# ============================================================
# 5. MODELO
# ============================================================

classifier = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=8,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


# ============================================================
# 6. PIPELINE COMPLETO
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", classifier),
    ]
)

pipeline.fit(X_train, y_train)


# ============================================================
# 7. EVALUACIÓN
# ============================================================

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
    "precision": round(
        float(precision_score(y_test, y_pred, zero_division=0)),
        6,
    ),
    "recall": round(
        float(recall_score(y_test, y_pred, zero_division=0)),
        6,
    ),
    "auc": round(float(roc_auc_score(y_test, y_proba)), 6),
    "threshold": UMBRAL,
}

print("\nMétricas del modelo:")
for nombre, valor in metrics.items():
    print(f"{nombre}: {valor}")


# ============================================================
# 8. EXTRACCIÓN DINÁMICA DE VALORES CATEGÓRICOS
# ============================================================

categorical_allowed_values = {
    column: sorted(
        df[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    for column in CATEGORICAL_COLUMNS
}


# ============================================================
# 9. BUNDLE DE PRODUCCIÓN
# ============================================================

bundle = {
    "model": pipeline,
    "target_column": TARGET,
    "column_names": FEATURE_COLUMNS.copy(),
    "numeric_columns": NUMERIC_COLUMNS.copy(),
    "categorical_columns": CATEGORICAL_COLUMNS.copy(),
    "categorical_allowed_values": categorical_allowed_values,
    "metrics": metrics,
    "threshold": UMBRAL,
    "training_datetime": datetime.now(timezone.utc),
    "random_state": RANDOM_STATE,
    "library_versions": {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "scikit_learn": sklearn.__version__,
        "joblib": version("joblib"),
    },
}


# ============================================================
# 10. GUARDADO DEL MODELO
# ============================================================

joblib.dump(
    bundle,
    ARCHIVO_MODELO,
    compress=3,
)


# ============================================================
# 11. VALIDACIÓN POST-SERIALIZACIÓN
# ============================================================

bundle_cargado = joblib.load(ARCHIVO_MODELO)

modelo_original = bundle["model"]
modelo_cargado = bundle_cargado["model"]

muestra_prueba = X_test.iloc[[0]].copy()

probabilidad_original = modelo_original.predict_proba(
    muestra_prueba
)[:, 1]

probabilidad_cargada = modelo_cargado.predict_proba(
    muestra_prueba
)[:, 1]

assert np.array_equal(
    probabilidad_original,
    probabilidad_cargada,
), (
    "ERROR: las probabilidades antes y después de "
    "serializar el modelo no coinciden exactamente."
)

assert (
    bundle_cargado["column_names"] == FEATURE_COLUMNS
), "ERROR: las columnas almacenadas no coinciden."

assert (
    bundle_cargado["categorical_allowed_values"]
    == categorical_allowed_values
), "ERROR: los valores categóricos almacenados no coinciden."


print("\nValidación de serialización: OK")
print(
    "Probabilidad original:",
    float(probabilidad_original[0]),
)
print(
    "Probabilidad cargada:",
    float(probabilidad_cargada[0]),
)
print("\nArchivos generados:")
print(f"- {ARCHIVO_MODELO}")
