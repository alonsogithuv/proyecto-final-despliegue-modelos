from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


TurnoOperativo = Literal["Día", "Tarde", "Noche"]
TipoMaquina = Literal["Prensa", "Torno", "CNC"]


class DatosManufactura(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        json_schema_extra={
            "example": {
                "temperatura": 78.5,
                "nivel_vibracion": 3.1,
                "presion_linea": 5.8,
                "velocidad_produccion": 235.0,
                "turno_operativo": "Noche",
                "tipo_maquina": "CNC",
            }
        },
    )

    temperatura: Annotated[
        float,
        Field(
            ...,
            strict=True,
            gt=0,
            le=150,
            description="Temperatura del proceso en grados Celsius.",
        ),
    ]

    nivel_vibracion: Annotated[
        float,
        Field(
            ...,
            strict=True,
            ge=0,
            le=10,
            description="Nivel de vibración de la máquina.",
        ),
    ]

    presion_linea: Annotated[
        float,
        Field(
            ...,
            strict=True,
            gt=0,
            le=20,
            description="Presión de la línea de producción.",
        ),
    ]

    velocidad_produccion: Annotated[
        float,
        Field(
            ...,
            strict=True,
            gt=0,
            le=500,
            description="Velocidad de producción en unidades por minuto.",
        ),
    ]

    turno_operativo: TurnoOperativo = Field(
        ...,
        description="Turno operativo de fabricación.",
    )

    tipo_maquina: TipoMaquina = Field(
        ...,
        description="Tipo de máquina utilizada.",
    )
