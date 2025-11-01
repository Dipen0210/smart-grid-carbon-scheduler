"""
Streamlit dashboard for the Smart Grid Carbon Reduction System (RL Edition).

Provides controls to interact with the FastAPI backend to fetch data, run demand
forecasts, and execute the RL-based carbon-aware scheduler while visualising the
outcomes with Plotly charts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import plotly.graph_objs as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Configure logging for debugging within Streamlit's runtime.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("streamlit-app")

# Directory to persist chart exports.
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


# ---------------------------------------------------------------------------
# Backend communication helpers
# ---------------------------------------------------------------------------

def build_url(base_url: str, endpoint: str) -> str:
    """Compose a full URL for the backend endpoint."""
    return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"


def request_backend(
    method: str,
    url: str,
    *,
    json_payload: Optional[Dict[str, Any]] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    """Perform an HTTP request with graceful error handling."""
    try:
        response = requests.request(method=method, url=url, json=json_payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Backend request failed: %s", exc)
        st.error(f"Request to {url} failed: {exc}")
        return {}


def read_csv_if_exists(path_str: Optional[str]) -> Optional[pd.DataFrame]:
    """Read a CSV path returned by the backend if the file exists."""
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        st.warning(f"File {path} not found on disk.")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to read {path}: {exc}")
        return None


def save_chart(fig: go.Figure, filename: str) -> Path:
    """Persist Plotly figure as HTML snapshot under frontend/output/."""
    output_path = OUTPUT_DIR / f"{filename}.html"
    fig.write_html(str(output_path))
    st.caption(f"Chart saved to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Streamlit layout helpers
# ---------------------------------------------------------------------------

def render_header() -> None:
    """Render the dashboard header."""
    st.title("⚡ Smart Grid Carbon Reduction System (RL Edition)")
    st.write(
        "Interact with the reinforcement learning powered backend to forecast demand, "
        "optimise energy schedules, and track carbon savings."
    )


def render_sidebar() -> tuple[Dict[str, Any], Dict[str, bool]]:
    """Render sidebar controls and return parameter values plus action triggers."""
    st.sidebar.header("⚙️ Scheduler Parameters")
    total_kwh = st.sidebar.number_input("Total Energy (kWh)", min_value=10.0, max_value=500.0, value=70.0)
    max_kw = st.sidebar.number_input("Max Power (kW)", min_value=1.0, max_value=20.0, value=7.2)
    hours = st.sidebar.number_input("Horizon (hours)", min_value=1, max_value=48, value=24, step=1)
    st.sidebar.markdown("---")
    st.sidebar.header("🗂 Actions")
    fetch_trigger = st.sidebar.button("📡 Fetch Latest Data", use_container_width=True)
    forecast_trigger = st.sidebar.button("🔮 Run Forecast", use_container_width=True)
    rl_trigger = st.sidebar.button("🧠 Run RL Scheduler", use_container_width=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("Run operations here; results stay visible in the main panel.")
    return (
        {
            "total_kwh": total_kwh,
            "max_kw": max_kw,
            "hours": int(hours),
        },
        {
            "fetch": fetch_trigger,
            "forecast": forecast_trigger,
            "rl": rl_trigger,
        },
    )


def initialize_session_state() -> None:
    """Ensure keys exist in Streamlit session state for persisted results."""
    defaults = {
        "fetch_result": None,
        "forecast_result": None,
        "rl_result": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def render_status(backend_url: str) -> None:
    """Display backend status information."""
    status_url = build_url(backend_url, "/status")
    status = request_backend("GET", status_url)
    if status:
        st.success(f"Backend status: {status.get('message', 'unknown')}")
    else:
        st.warning("Unable to reach backend status endpoint.")


def perform_fetch_data(backend_url: str) -> Optional[Dict[str, Any]]:
    """Trigger the fetch-data workflow and return the backend response plus dataset."""
    fetch_url = build_url(backend_url, "/fetch-data")
    response = request_backend("GET", fetch_url)
    if not response:
        return None
    dataset = read_csv_if_exists(response.get("path"))
    return {
        "response": response,
        "dataset": dataset,
    }


def render_fetch_result(result: Dict[str, Any]) -> None:
    """Render the outcome of a fetch-data run."""
    response = result.get("response")
    if not response:
        return
    st.success(response.get("message", "Fetch completed."))
    st.json(response)
    metadata = response.get("metadata") or {}
    label = metadata.get("label")
    if label == "year":
        st.info("Baseline 1-year datasets captured. Copies also saved as latest aliases for initial training.")
    elif label == "latest":
        st.info("Weekly refresh completed. Downstream components now use the freshest 7-day window.")
    dataset = result.get("dataset")
    if isinstance(dataset, pd.DataFrame):
        st.dataframe(dataset.tail(10), use_container_width=True)


def perform_forecast(backend_url: str) -> Optional[Dict[str, Any]]:
    """Run demand forecast and return both the backend response and dataframe."""
    forecast_url = build_url(backend_url, "/forecast-demand")
    response = request_backend("POST", forecast_url)
    if not response:
        return None
    dataframe = read_csv_if_exists(response.get("path"))
    return {
        "response": response,
        "dataframe": dataframe,
    }


def render_forecast_result(result: Dict[str, Any], *, save_chart_output: bool) -> None:
    """Visualise the forecast result and optionally persist the chart."""
    response = result.get("response")
    if not response:
        return
    st.success(response.get("message", "Forecast generated."))
    st.json(response)
    metadata = response.get("metadata") or {}
    history_start = metadata.get("history_start")
    history_end = metadata.get("history_end")
    if history_start and history_end:
        st.info(
            f"Forecast trained on demand history from {history_start} to {history_end} (UTC)."
        )
    dataframe = result.get("dataframe")
    if not isinstance(dataframe, pd.DataFrame):
        st.warning("Forecast data unavailable.")
        return

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=pd.to_datetime(dataframe["timestamp"]),
            y=dataframe["forecast_mw"],
            mode="lines+markers",
            name="Forecast MW",
        )
    )
    figure.update_layout(
        title="24h Demand Forecast",
        xaxis_title="Time (UTC)",
        yaxis_title="Demand (MW)",
        template="plotly_dark",
    )
    st.plotly_chart(figure, use_container_width=True)
    if save_chart_output:
        save_chart(figure, "demand_forecast")
    st.info(
        f"Forecast includes {len(dataframe)} points spanning "
        f"{dataframe['timestamp'].min()} to {dataframe['timestamp'].max()}."
    )


def perform_run_rl(
    backend_url: str,
    *,
    total_kwh: float,
    max_kw: float,
    hours: int,
) -> Optional[Dict[str, Any]]:
    """Execute the RL scheduler and return the backend response plus schedule dataframe."""
    rl_url = build_url(backend_url, "/run-rl")
    payload = {
        "total_kwh": total_kwh,
        "max_kw": max_kw,
        "hours": hours,
    }
    response = request_backend("POST", rl_url, json_payload=payload)
    if not response:
        return None

    schedule_records = response.get("schedule") or []
    dataframe = pd.DataFrame(schedule_records) if schedule_records else None
    return {
        "response": response,
        "dataframe": dataframe,
    }


def render_rl_result(result: Dict[str, Any], *, save_chart_output: bool) -> None:
    """Render RL scheduling outputs including metrics and charts."""
    response = result.get("response")
    if not response:
        return
    st.success(response.get("message", "RL scheduling completed."))
    st.json(response)

    dataframe = result.get("dataframe")
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        st.warning("No schedule data returned from backend.")
        return

    st.dataframe(dataframe, use_container_width=True)

    metadata = response.get("metadata") or {}
    carbon_saving_percent = float(metadata.get("carbon_saving_percent", response.get("carbon_saving_percent", 0.0)))
    total_emissions = float(metadata.get("total_emissions_kg", response.get("total_emissions_kg", 0.0)))
    energy_target_met = float(metadata.get("energy_target_met_percent", response.get("energy_target_met_percent", 0.0)))

    met_col, co2_col, energy_col = st.columns(3)
    with co2_col:
        st.metric(label="Total CO₂ (kg)", value=f"{total_emissions:.2f}")
    with met_col:
        st.metric(label="CO₂ Reduction", value=f"{carbon_saving_percent:.2f}%")
    with energy_col:
        st.metric(label="Energy Target Met", value=f"{energy_target_met:.2f}%")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dataframe["hour"],
            y=dataframe["energy_kwh"],
            name="Energy Dispatch (kWh)",
            marker_color="#1f77b4",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dataframe["hour"],
            y=dataframe["carbon_intensity_gco2_per_kwh"],
            mode="lines+markers",
            name="Carbon Intensity (gCO₂/kWh)",
            line=dict(color="#ff7f0e"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="RL Schedule vs Carbon Intensity",
        xaxis_title="Hour",
        yaxis=dict(title="Energy (kWh)", side="left"),
        yaxis2=dict(title="Carbon Intensity (gCO₂/kWh)", overlaying="y", side="right"),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)
    if save_chart_output:
        save_chart(fig, "rl_schedule")

    dispatched = dataframe["energy_kwh"].sum()
    st.info(f"RL scheduler dispatched {dispatched:.2f} kWh across {len(dataframe)} hours.")


# ---------------------------------------------------------------------------
# Main Streamlit application flow
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the Streamlit app."""
    render_header()
    initialize_session_state()
    user_inputs, actions = render_sidebar()
    backend_url = DEFAULT_BACKEND_URL

    # Display backend status on every page load.
    render_status(backend_url)

    triggered_action: Optional[str] = None

    if actions["fetch"]:
        result = perform_fetch_data(backend_url)
        if result is not None:
            st.session_state["fetch_result"] = result
        triggered_action = "fetch"

    if actions["forecast"]:
        result = perform_forecast(backend_url)
        if result is not None:
            st.session_state["forecast_result"] = result
        triggered_action = "forecast"

    if actions["rl"]:
        result = perform_run_rl(
            backend_url,
            total_kwh=user_inputs["total_kwh"],
            max_kw=user_inputs["max_kw"],
            hours=user_inputs["hours"],
        )
        if result is not None:
            st.session_state["rl_result"] = result
        triggered_action = "rl"

    st.subheader("Results")
    sections_rendered = 0

    fetch_result = st.session_state.get("fetch_result")
    if fetch_result:
        sections_rendered += 1
        st.markdown("### 📡 Latest Data Fetch")
        render_fetch_result(fetch_result)

    forecast_result = st.session_state.get("forecast_result")
    if forecast_result:
        if sections_rendered:
            st.markdown("---")
        sections_rendered += 1
        st.markdown("### 🔮 Demand Forecast")
        render_forecast_result(
            forecast_result,
            save_chart_output=triggered_action == "forecast",
        )

    rl_result = st.session_state.get("rl_result")
    if rl_result:
        if sections_rendered:
            st.markdown("---")
        sections_rendered += 1
        st.markdown("### 🧠 RL Scheduler")
        render_rl_result(
            rl_result,
            save_chart_output=triggered_action == "rl",
        )

    if not sections_rendered:
        st.info("Use the sidebar to fetch data, run forecasts, or execute the RL scheduler.")

    st.markdown("---")
    st.caption("Need API details? Visit the FastAPI documentation served by the backend at `/docs`.")


if __name__ == "__main__":
    main()
