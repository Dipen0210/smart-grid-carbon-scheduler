from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from backend import config
from backend.schemas import (
    ForecastRequest,
    OperationResponse,
    RLRequest,
    RLScheduleResponse,
)
from backend.services import data_service, forecast_service, rl_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("backend")

app = FastAPI(
    title="Smart Grid Carbon Reduction System (Reinforcement Learning Edition)",
    version="1.0.0",
    description=(
        "APIs for data collection, demand forecasting, and reinforcement learning based"
        " scheduling to minimise carbon emissions."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    config.ensure_processed_dir()
    logger.info("Processed data directory ensured at %s.", config.PROCESSED_DATA_DIR)


@app.get("/status", response_model=OperationResponse)
async def status() -> OperationResponse:
    logger.debug("Status endpoint invoked.")
    return OperationResponse(message="Backend is running.")


@app.get("/fetch-data", response_model=OperationResponse)
async def fetch_data() -> OperationResponse:
    logger.info("Fetch-data endpoint triggered.")
    try:
        path, metadata = await run_in_threadpool(data_service.fetch_fresh_data)
        return OperationResponse(
            message="Fetched and stored the latest energy datasets.",
            path=str(path),
            metadata=metadata,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Data fetch failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/forecast-demand", response_model=OperationResponse)
async def forecast_demand(
    request: ForecastRequest = ForecastRequest(),
) -> OperationResponse:
    logger.info(
        "Forecast-demand endpoint triggered for horizon %d hours.",
        request.horizon_hours,
    )
    try:
        result = await run_in_threadpool(
            forecast_service.run_demand_forecast,
            request.resolve_data_path(),
            request.horizon_hours,
        )
        return OperationResponse(
            message="Demand forecast generated successfully.",
            path=str(result.path),
            metadata={
                "history_start": result.history_start.isoformat(),
                "history_end": result.history_end.isoformat(),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Forecast generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/run-rl", response_model=RLScheduleResponse)
async def run_rl(request: RLRequest) -> RLScheduleResponse:
    logger.info(
        "Run-rl endpoint triggered: total_kwh=%.2f, max_kw=%.2f, hours=%d.",
        request.total_kwh,
        request.max_kw,
        request.hours,
    )
    try:
        path, schedule_records, metadata = await run_in_threadpool(
            rl_service.run_rl_scheduler,
            request.total_kwh,
            request.max_kw,
            request.hours,
            request.resolve_carbon_path(),
            request.episodes,
        )
        carbon_saving = float(metadata.get("carbon_saving_percent", 0.0))
        total_emissions = float(metadata.get("total_emissions_kg", 0.0))
        total_energy = float(metadata.get("total_energy_scheduled_kwh", 0.0))
        energy_target_met = float(metadata.get("energy_target_met_percent", 0.0))
        return RLScheduleResponse(
            message="RL scheduling completed successfully.",
            path=str(path),
            carbon_saving_percent=carbon_saving,
            total_emissions_kg=total_emissions,
            total_energy_scheduled_kwh=total_energy,
            energy_target_met_percent=energy_target_met,
            schedule=schedule_records,
            metadata=metadata,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("RL scheduling failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def get_app() -> FastAPI:
    """Convenience accessor for ASGI servers."""
    return app
