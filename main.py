from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from esquemas import DatosManufactura


MODEL_PATH = Path("modelo_produccion.joblib")

bundle: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bundle

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo de producción: {MODEL_PATH}"
        )

    bundle = joblib.load(MODEL_PATH)

    required_keys = {
        "model",
        "target_column",
        "column_names",
        "numeric_columns",
        "categorical_columns",
        "categorical_allowed_values",
        "metrics",
        "threshold",
        "training_datetime",
        "library_versions",
    }

    missing_keys = required_keys.difference(bundle.keys())

    if missing_keys:
        raise RuntimeError(
            f"Bundle inválido. Faltan las claves: {sorted(missing_keys)}"
        )

    yield

    bundle.clear()


app = FastAPI(
    title="Manufacturing Defect Prediction API",
    description="API para clasificación binaria de defectos de manufactura.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def health_check() -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "manufacturing-defect-prediction",
        "model_loaded": bool(bundle),
        "target_column": bundle.get("target_column"),
        "metrics": bundle.get("metrics"),
        "threshold": bundle.get("threshold"),
        "training_datetime": (
            bundle["training_datetime"].isoformat()
            if hasattr(bundle.get("training_datetime"), "isoformat")
            else bundle.get("training_datetime")
        ),
        "library_versions": bundle.get("library_versions"),
    }


@app.post("/predecir")
async def predecir(datos: DatosManufactura) -> dict[str, Any]:
    if not bundle:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está cargado.",
        )

    model = bundle["model"]
    column_names = bundle["column_names"]
    threshold = float(bundle["threshold"])

    input_data = datos.model_dump()

    dataframe = pd.DataFrame(
        [input_data],
        columns=column_names,
    )

    try:
        probability = float(model.predict_proba(dataframe)[0, 1])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante la predicción: {exc}",
        ) from exc

    predicted_label = int(probability >= threshold)

    return {
        "probabilidad_defecto": round(probability, 6),
        "etiqueta_predictiva": predicted_label,
        "resultado": (
            "DEFECTO_DETECTADO"
            if predicted_label == 1
            else "SIN_DEFECTO_DETECTADO"
        ),
        "umbral_utilizado": threshold,
    }
