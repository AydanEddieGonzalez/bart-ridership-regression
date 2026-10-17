import pandas as pd
import requests

# Downtown Oakland, roughly central to the BART system
params = {
    "latitude": 37.80,
    "longitude": -122.27,
    "start_date": "2024-01-01",
    "end_date": "2025-12-31",
    "daily": "temperature_2m_max,precipitation_sum",
    "temperature_unit": "fahrenheit",
    "precipitation_unit": "inch",
    "timezone": "America/Los_Angeles",
}

resp = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params)
resp.raise_for_status()

weather = pd.DataFrame(resp.json()["daily"]).rename(columns={
    "time": "date",
    "temperature_2m_max": "temp_max_f",
    "precipitation_sum": "precip_in",
})
weather.to_csv("data/raw/weather.csv", index=False)
print(weather.head(), f"\n{len(weather)} days saved")