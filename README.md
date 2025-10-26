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

### 🏠 Example Real-World Scenario

**User Context:**  
A user has a **smart home** connected to the grid through this AI-powered system.  
Their home includes an **EV charger**, a **washing machine**, and other smart appliances.

---

**User Goal:**  
The user wants all daily tasks — EV charging, laundry, and appliance use — to complete before morning,  
but with **minimum carbon emissions** and **no manual control**.

---

**System Input (through dashboard or app):**
- Energy Needed: **70 kWh**
- Time Window: **9 PM → 9 AM**
- Devices: EV charger, washing machine, heater
- Forecast Model: **LSTM**

---

**How the System Works:**
1. The AI model forecasts renewable generation and grid demand for the next 24 hours.  
2. It fetches live carbon-intensity data from the national grid (CO₂ per kWh).  
3. Within the user’s 12-hour window, it finds the *cleanest hours* for consumption.  
4. It automatically schedules and controls smart devices accordingly:
   - 🧺 **Washing machine** starts automatically during the cleanest hour (e.g., 1 AM).  
   - 🚗 **EV charger** adjusts its charging speed:
     - Slower during fossil-heavy hours (e.g., 9–11 PM)  
     - Faster when renewable power peaks (e.g., 2–4 AM)  
   - 🔥 **Water heater** runs only when clean energy is available.  
5. All tasks finish by 9 AM — without user intervention.

---

**System Output:**
- ⚡ Baseline Emissions: **4.37 kg CO₂e**
- 🌱 Optimized Emissions: **3.43 kg CO₂e**
- 💡 Total Reduction: **~21%**
- 🕐 Smart Schedule:
  - EV Charging: **11 PM – 3 AM**
  - Laundry: **1 AM – 2 AM**
  - Heating: **2 AM – 4 AM**

---

**Result:**
✅ All devices finish tasks on time  
✅ User comfort unchanged  
✅ Grid demand balanced  
✅ CO₂ emissions significantly reduced  

> The user doesn’t have to do anything — the AI automatically optimizes all smart-home energy tasks for the lowest-carbon hours while meeting every deadline.


---

### Summary  
- Uses AI + real-time carbon data to make electricity consumption smarter, cleaner, and more sustainable.  
- Inspired by carbon-aware systems at Google, Microsoft, and Tesla.
