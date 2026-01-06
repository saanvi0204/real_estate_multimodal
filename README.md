# Multimodal Real Estate Price Prediction using Satellite Imagery

This project develops a **multimodal regression pipeline** to predict residential property prices by combining **tabular housing attributes** with **satellite imagery**. The approach integrates environmental context—such as green cover, road density, and proximity to water—into traditional real estate valuation models.

The project follows a **research-oriented workflow**, progressing from exploratory analysis to multimodal modeling, explainability, and final inference.

---

## Project Highlights

- Combines **tabular data + satellite images** for price prediction  
- Uses **residual learning** to let images correct tabular model errors  
- Evaluates multiple **multimodal fusion strategies**  
- Provides **model explainability** via Grad-CAM  
- Produces a final **prediction CSV** for unseen test data  

---

## Repository Structure

```

real_estate_multimodal/
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── image_embedding_extraction.ipynb
│   ├── geospatial_visual_eda.ipynb
│   ├── model_training.ipynb
│   ├── grad_cam.ipynb
│   └── final_prediction.ipynb
│
├── src/
│   └── data_fetcher.py
│
├── models/
│   ├── tabular_xgb.json
│   ├── tabular_scaler.joblib
│   ├── residual_cnn.pt
│
├── outputs/
│   ├── 23323035_final.csv
│   └── gradcam/
│
├── requirements.txt
├── README.md
└── 23323035_report.pdf

````

---

## Setup Instructions

### Create Environment

Using **Anaconda** (recommended):

```bash
conda create -n real_estate_mm python=3.11
conda activate real_estate_mm
pip install -r requirements.txt
````

---

### API Key Configuration (Satellite Images)

Satellite images are fetched programmatically using a map API (e.g., Google Maps Static API).

1. Create a `.env` file in the project root:

```
MAPS_API_KEY=your_api_key_here
```

2. The key is automatically read inside `src/data_fetcher.py`.

**Do not commit `.env` to GitHub** (already ignored via `.gitignore`).

---

## Satellite Image Acquisition (`data_fetcher.py`)

The script `src/data_fetcher.py` is responsible for **programmatically downloading satellite images** for each property using its latitude and longitude.

### Purpose

* Augment tabular housing data with **visual environmental context**
* Capture features such as:

  * Green cover
  * Road density
  * Urban vs residential texture
  * Proximity to water bodies

### Functionality

`data_fetcher.py` performs the following steps:

1. **Reads property IDs and coordinates** (latitude, longitude) from the dataset
2. **Constructs API requests** to a satellite imagery service (e.g., Google Maps Static API)
3. **Downloads satellite images** at a fixed zoom level and resolution
4. **Saves images locally** using the convention:

```
data/images/<property_id>.0.png
```

5. **Implements safeguards**:

   * Skips images that already exist
   * Applies rate-limiting between API calls
   * Logs failed downloads without interrupting execution

This script must be executed **before any image-based modeling steps**.

---

## Notebook Execution Order (IMPORTANT)

Run notebooks **in the following order** to reproduce results:

```
1. src/data_fetcher.py
2. notebooks/preprocessing.ipynb
3. notebooks/geospatial_visual_eda.ipynb
4. notebooks/image_embedding_extraction.ipynb
5. notebooks/model_training.ipynb
6. notebooks/grad_cam.ipynb
7. notebooks/final_prediction.ipynb
```

---

## Notebook Descriptions

### `preprocessing.ipynb`

* Cleans raw housing data
* Engineers tabular features
* Applies log-price transformation
* Outputs cleaned dataset

---

### `geospatial_visual_eda.ipynb`

* Geospatial analysis using latitude/longitude
* Price heatmaps and clustering
* Visual inspection of satellite images
* Identifies environmental pricing patterns

---

### `image_embedding_extraction.ipynb`

* Trains a **residual CNN (ResNet-18)** on satellite images
* CNN learns to predict tabular model residuals
* Saves trained image model and embeddings

---

### `model_training.ipynb`

* Evaluates multimodal fusion strategies:

  * Tabular baseline
  * Late fusion
  * Early fusion
  * Hybrid (tree-based)
  * Residual fusion
  * Stacked generalization
* Compares models using **RMSE and R²**
* Selects final model

---

### `grad_cam.ipynb`

* Applies **Grad-CAM** to the trained CNN
* Visualizes influential image regions
* Validates that the model focuses on meaningful environmental features

---

### `final_prediction.ipynb`

* Runs inference on test data
* Loads trained tabular + image models
* Generates final price predictions
* Saves output to `outputs/predictions.csv`

---

## Reproducibility Notes

* All random seeds are fixed where applicable
* Pretrained ImageNet weights are used for CNN initialization
* Tabular scaler and models are saved and reused (no test leakage)
* Satellite image resolution and zoom are consistent across train and test
* Processed datasets and intermediate artifacts are generated during execution and not committed
* No retraining is performed during inference

---

## Final Output

The final deliverable is:

```
outputs/23323035_final.csv
```

Format:

```
id,predicted_price
```

This file contains predicted property prices for the unseen test dataset.

---

## Evaluation Metrics

Models are evaluated using:

* **RMSE (Root Mean Squared Error)**
* **R² Score**

Comparisons between tabular-only and multimodal models are reported in `model_training.ipynb`.

---

## Key Takeaway

This project demonstrates that **satellite imagery provides complementary environmental signals** that improve real estate valuation when integrated thoughtfully with tabular data—particularly via residual learning and explainable deep vision models.

---

## Notes

* The project is structured for **clarity, reproducibility, and interpretability**
* Each notebook has a clearly scoped responsibility
* Designed to reflect real-world ML workflows and research practices
