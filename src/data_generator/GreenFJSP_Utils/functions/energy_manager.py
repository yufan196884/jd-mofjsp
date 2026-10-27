import warnings
from pathlib import Path

from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pvlib.iotools import get_pvgis_hourly


class EnergyManager:
    # ------------------------------------------------------------------ 
    #  Solar constants 
    # ------------------------------------------------------------------ 
    _YEAR_SOLAR       = 2019
    _PEAK_POWER_KWP   = 186.0
    _SYSTEM_LOSS      = 14
    _TILT             = 10
    _LATITUDE         = 51.32
    _LONGITUDE        = 7.00
    _AZIMUTH          = 172
    _YEAR_PRICES      = 2025

    def __init__(self, month: int):
        self.target_month = month

        # Precompute interpolation indices 
        self._old_idx = np.linspace(0, 1439, 1440)
        self._new_idx = np.linspace(0, 1439, 2400)

        csv_path = Path(__file__).parent.parent / "data" / "Spotmarktpreis.csv"

        # Load everything once 
        self._solar_profiles: list[np.ndarray] = self._load_solar_profiles()
        self._price_profiles: list[np.ndarray] = self._load_price_profiles(csv_path)


    def get_solar_energy(self) -> np.ndarray:
        """Return a randomly sampled day profile (1440 values, kW)."""
        return self._solar_profiles[np.random.randint(len(self._solar_profiles))].copy()

    def get_energy_prices(self) -> np.ndarray:
        """Return a randomly sampled day price profile (1440 values, ct/kWh)."""
        return self._price_profiles[np.random.randint(len(self._price_profiles))].copy()

    def convert_minutes_to_industrial_minutes(self, minutes_array: np.ndarray) -> np.ndarray:
        """Resample a 1440-length array to 2400 industrial minutes."""
        if len(minutes_array) != 1440:
            raise ValueError(f"Expected length 1440, got {len(minutes_array)}")
        return np.interp(self._new_idx, self._old_idx, minutes_array)


    def _load_solar_profiles(self) -> List[np.ndarray]:
        """Fetch one year of PVGIS data and split into daily profiles."""
        try:
            data = get_pvgis_hourly(
                latitude       = self._LATITUDE,
                longitude      = self._LONGITUDE,
                pvcalculation  = True,
                start          = self._YEAR_SOLAR,
                end            = self._YEAR_SOLAR,
                raddatabase    = "PVGIS-SARAH3",
                surface_tilt   = self._TILT,
                surface_azimuth= self._AZIMUTH,
                peakpower      = self._PEAK_POWER_KWP,
                loss           = self._SYSTEM_LOSS,
                outputformat   = "json",
            )
            data = data[0]
        except Exception as exc:
            warnings.warn(f"PVGIS API error: {exc}. Using fallback sine profiles.")
            return self._generate_fallback_solar()

        df = data[data.index.month == self.target_month].copy()
        df["power_kw"] = df["P"] / 1000.0
        df_min = df["power_kw"].resample("1min").interpolate(method="linear")

        profiles = []
        for _, group in df_min.groupby(df_min.index.date):
            if len(group) == 1440:
                profile = group.values.copy()
                profile[profile < 0] = 0.0
                profiles.append(profile)

        if not profiles:
            warnings.warn("No complete daily profiles found in PVGIS data. Using fallback.")
            return self._generate_fallback_solar()

        return profiles

    def _load_price_profiles(self, csv_path: Path) -> List[np.ndarray]:
        """Load the price CSV once and split into per-day profiles."""
        df = pd.read_csv(csv_path, sep=";", decimal=",")
        df["timestamp"] = pd.to_datetime(
            df["Datum"] + " " + df["von"], format="%d.%m.%Y %H:%M"
        )
        df_filtered = df[
            (df["timestamp"].dt.year  == self._YEAR_PRICES) &
            (df["timestamp"].dt.month == self.target_month)
        ]
        if df_filtered.empty:
            raise ValueError(
                f"No price data for month {self.target_month}/{self._YEAR_PRICES} "
                f"in {csv_path}."
            )

        profiles = []
        for day, group in df_filtered.groupby(df_filtered["timestamp"].dt.date):
            group = group.sort_values("timestamp")
            prices = group["Spotmarktpreis in ct/kWh"].values

            # Normalise to exactly 24 hourly values
            if len(prices) != 24:
                prices = np.interp(
                    np.linspace(0, 24, 24),
                    np.arange(len(prices)),
                    prices,
                )

            profiles.append(np.repeat(prices, 60))  # 1440 values

        return profiles

    # ------------------------------------------------------------------ #
    #  Fallback                                                            
    # ------------------------------------------------------------------ #

    def _generate_fallback_solar(self) -> List[np.ndarray]:
        """30 slightly varied sine-wave day profiles."""
        x = np.linspace(0, np.pi, 1440)
        base = np.sin(x)
        profiles = []
        for _ in range(30):
            cloud  = np.random.uniform(0.2, 1.0)
            noise  = np.random.normal(0.0, 0.05, 1440)
            curve  = base * self._PEAK_POWER_KWP * cloud + noise
            curve[curve < 0] = 0.0
            profiles.append(curve)
        return profiles

    # ------------------------------------------------------------------ #
    #  Visualisation                                                       
    # ------------------------------------------------------------------ #

    def plot(self, solar: np.ndarray, prices: np.ndarray, IM: bool = True) -> None:
        n = 2400 if IM else 1440
        x = np.linspace(0, 24, n)

        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_xlim([0, 24])
        ax1.set_xlabel("Time [h]", fontsize=12)
        ax1.set_ylabel("Solar generation [kW]", fontsize=12)
        ln1 = ax1.plot(x, solar, label="Solar generation")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Day-ahead price [ct/kWh]", fontsize=12)
        ln2 = ax2.plot(x, prices, linestyle="--", alpha=0.7, label="Electricity price")
        ax2.axhline(0, color="black", linewidth=0.8, linestyle=":")

        lns  = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="upper right")
        plt.title(f"Solar profile vs. day-ahead prices (month: {self.target_month})")
        plt.tight_layout()
        suffix = "_IM" if IM else ""
        plt.savefig(f"solar_vs_prices{suffix}.png")


# ---------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    em = EnergyManager(month=6)          

    solar  = em.get_solar_energy()       
    prices = em.get_energy_prices()      

    solar_im  = em.convert_minutes_to_industrial_minutes(solar)
    prices_im = em.convert_minutes_to_industrial_minutes(prices)

    em.plot(solar,    prices,    IM=False)
    em.plot(solar_im, prices_im, IM=True)
    

    print("Done.")
    print(f"Solar  length : {len(solar_im)}")
    print(f"Prices length : {len(prices_im)}")