# Flood Image + Environmental Sensor Mini‑Dataset

## 1. Purpose

This mini‑dataset is meant for **quick‑turn prototyping and validation** of multimodal flood‑detection pipelines that fuse visual cues with basic atmospheric telemetry (relative humidity, air temperature, sea‑level pressure).

## 2. Contents

```text
 test_dataset/
 ├── images/               # 20 JPEGs (10 flood, 10 no‑flood)
 │   ├── flood_1.jpg … flood_10.jpg
 │   └── no_flood_1.jpg … no_flood_10.jpg
 ├── labels/               # one JSON per image
 │   ├── flood_1.json … flood_10.json
 │   └── no_flood_1.json … no_flood_10.json
 └── README.md             # this file
```

Each JSON includes:

```json5
{
  "image": "flood_7.jpg",        // file name inside images/
  "label": "flood",              // or "no_flood"
  "flood_level": "medium-severe",// only for flood samples
  "sensor_data": {
    "humidity": 88,              // %RH
    "temperature": 18.7,         // °C
    "pressure": 1003             // hPa
  }
}
```

## 3. Baseline (Non‑Flood) Sensor Profile

| Variable           | Baseline value | Units | Notes                                |
| ------------------ | -------------- | ----- | ------------------------------------ |
| Relative humidity  | **50 %**       |  %    | Mid‑latitude fair‑weather afternoon  |
| Temperature        | **22 °C**      | °C    | Neutral thermal comfort (ASHRAE 55)  |
| Sea‑level pressure | **1015 hPa**   | hPa   | Mean sea‑level pressure used by NOAA |

Gaussian measurement noise (σ≈ ±5 %RH, ±1 °C, ±2 hPa) is applied to every non‑flood sample to mimic sensor jitter.

## 4. Flood‑Level Taxonomy & Expected Sensor Shifts

| Flood level       | Qualitative description                     | Target RH (%) | ΔT (°C) from 22 °C | ΔP (hPa) from 1015 | Example file(s)      |
| ----------------- | ------------------------------------------- | ------------- | ------------------ | ------------------ | -------------------- |
| **little**        | Minor pooling / wet patches                 | 70–80         |  −1 – −2           |  −5 – −7           | flood\_1, flood\_10  |
| **medium**        | Roadway partially submerged                 |  80–85        |  −2 – −3           |  −8 – −10          | flood\_2             |
| **medium‑severe** | Extensive coverage, fast runoff             |  85–90        |  −2.5 – −3.5       |  −9 – −11          | flood\_5, flood\_7   |
| **severe**        | Widespread flooding, flash‑flood conditions |  90–95        |  −3 – −4           |  −11 – −17         | flood\_3, 4, 6, 8, 9 |

These ranges are distilled from peer‑reviewed sensor studies of flash floods, atmospheric‑river coastal floods, and hurricane landfalls. See **Section 7 References**.

## 5. Data‑Generation Workflow

1. **Image Selection** – 10 clear flood photos and 10 dry/no‑flood photos were drawn from public datasets (Kaggle, Roboflow) and manually vetted.
2. **Flood‑Level Labeling** – The user assigned a qualitative level (little, medium, medium‑severe, severe) to each flood image.
3. **Sensor Synthesis**

   * **No‑flood samples:** baseline + Gaussian noise.
   * **Flood samples:** baseline + deterministic level‑specific deltas sampled uniformly within the bounds in Section 4, then Gaussian noise (σ≈ ±2 %RH, ±0.3 °C, ±1 hPa) added for realism.
4. **File Writing** – Each image’s telemetry is written to `sensor/flood_<n>.json` (or `no_flood_<n>.json`) with the schema shown above.

## 6. Key References

1. Gochis et al. (2017) – Blanco River flash‑flood hydrometeorology.
2. Acosta‑Coll et al. (2018) – Urban flash‑flood wireless sensor thresholds.
3. Piecuch et al. (2022) – Atmospheric‑river surges and barometric setup.
4. Zhu et al. (2024) – 95 % RH precursor to extreme rainfall in Nanjing.
5. EPA Indoor Air Guidance – Baseline 30–50 % RH.
6. ASHRAE‑55 (2017) – Neutral comfort band 22–26 °C.

---

**Last updated:** 2 May 2025

