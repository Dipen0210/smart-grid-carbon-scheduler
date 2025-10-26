# ⚡ Smart Grid Carbon Reduction System

### Purpose  
An AI-powered system that forecasts renewable energy (solar & wind) and grid demand, retrieves real-time carbon intensity data, and automatically schedules energy consumption (like EV charging) during cleaner hours — reducing CO₂ emissions without affecting user comfort.

---

### Technologies Used  
- **Programming:** Python  
- **Data Processing:** Pandas, NumPy  
- **Machine Learning:** PyTorch (LSTM, Transformer models)  
- **API & Backend:** FastAPI, Requests  
- **Frontend & Visualization:** Streamlit, Matplotlib, Plotly  
- **Testing:** Pytest  
- **Datasets:** NREL (solar/wind), PJM Hourly Demand, UK Carbon Intensity API  

---

### Key Features  
- 📈 **Forecasting:** Predicts next 24 hours of renewable generation & demand.  
- 🌍 **Carbon Awareness:** Fetches real CO₂ intensity from grid APIs.  
- ⚙️ **Optimization:** Chooses lowest-carbon hours for energy consumption.  
- 🧠 **Automation:** Runs automatically — no manual user action needed.  
- 🖥️ **Visualization:** Streamlit dashboard shows forecasts, clean-hour schedules, and CO₂ reduction results.

---

### How It Works  
1. **Data Collection:** Load solar, weather, and demand datasets.  
2. **Forecasting:** AI models (LSTM/Transformer) predict renewable generation and demand for the next 24 hours.  
3. **Carbon Awareness:** Retrieve real-time carbon intensity data from the grid API.  
4. **Optimization:** Schedule energy tasks (e.g., 70 kWh over 12 hrs) in the lowest-carbon hours.  
5. **Visualization:** Show forecast, optimized schedule, and emission reduction results.

---

### Example Output  
- **Baseline Emissions:** 4.37 kg CO₂e  
- **Optimized Emissions:** 3.43 kg CO₂e  
- **Reduction:** ~21 %

---

### Summary  
> Uses AI + real-time carbon data to make electricity consumption smarter, cleaner, and more sustainable.  
> Inspired by carbon-aware systems at Google, Microsoft, and Tesla.
