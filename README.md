# Wound Segmentation and LLM-Enabled Clinical Assistant

**Deep learning-based wound segmentation, image-derived wound characterization, and an interactive LLM assistant in a Streamlit application.**

This research prototype combines a trained Keras segmentation model with classical image analysis and a conversational AI interface. Users upload a wound photograph, inspect the predicted wound region, review quantitative image features, and ask questions about those features through an LLM-powered assistant.

> **Research and educational use only.** This application is not a validated medical device. Its measurements, heuristic severity score, and generated responses are not diagnoses, clinical recommendations, or substitutes for professional assessment.

## Features

- **Image upload:** Accepts JPG, JPEG, and PNG images through Streamlit.
- **Automated segmentation:** Preprocesses images to 256 × 256 pixels and runs a saved Keras segmentation model; a 0.5 threshold converts probabilities to a binary mask.
- **Visual results:** Shows the uploaded image and a color overlay with bounding boxes around detected regions.
- **Image-derived measurements:** Reports region area (pixels), perimeter, circularity, color, texture contrast/homogeneity, and edge sharpness. Multiple contours are summarized with aggregate measurements.
- **Experimental severity index:** Computes a heuristic score from contrast, edge strength, and circularity. **The score has not been established as a clinically validated wound-severity measure.**
- **Context-aware chat:** Passes calculated wound properties into a chat conversation, allowing questions about the analysis.

## System workflow

```text
Upload wound image (JPG / JPEG / PNG)
                |
                v
     Resize to 256 x 256
       and normalize
                |
                v
    Keras segmentation model
                |
                v
      Binary mask (> 0.5)
                |
         +------+------+
         |             |
         v             v
 Overlay and       Quantitative
 bounding boxes    image features
         |             |
         +------+------+
                |
                v
     Streamlit result display
                |
                v
  LLM chat using wound features
                |
                v
   Conversational interpretation
```

## Technology stack

| Component | Technology |
| --- | --- |
| Application interface | Streamlit |
| Segmentation inference | TensorFlow / Keras |
| Image processing | OpenCV, Pillow, NumPy |
| Texture features | scikit-image (GLCM) |
| LLM access | OpenRouter API (`openai/gpt-4o-mini` in the current code) |
| API requests | Python `requests` |

## Repository structure

```text
.
├── ChatAPI.py               # LLM API integration and chat interface
├── WoundDetection.py        # Streamlit app, inference, features, visualization
├── wound-detection.ipynb    # Wound detection/segmentation notebook
├── eval_analysis.ipynb      # Evaluation and analysis notebook
├── models/                  # Model-related directory
└── README.md
```

**Model checkpoint:** `WoundDetection.py` currently loads `best_wound_unet_model.keras` from the repository's working directory. Ensure that the trained checkpoint is present at that exact path before starting the application. The public repository structure alone does not establish that this checkpoint is available at the expected location.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/fisha-mehabaw/wound-segmentation-and-LLM-enabled-clinical-assistant.git
cd wound-segmentation-and-LLM-enabled-clinical-assistant
```

### 2. Create a Python environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 3. Install required packages

```bash
pip install streamlit tensorflow numpy opencv-python pillow scikit-image requests
```

These packages are inferred from the imports in the application files. The repository does not currently include a pinned `requirements.txt`; dependency compatibility may vary by Python and TensorFlow version.

### 4. Configure the model and API credentials

1. Put the trained Keras checkpoint at `./best_wound_unet_model.keras`, or update the `load_model(...)` path in `WoundDetection.py` to point to its actual location.
2. **Before running or sharing the application, remove the hard-coded OpenRouter API key from `ChatAPI.py` and revoke the exposed key.** Use a new credential stored outside source control.
3. For example, change the API-key assignment in `ChatAPI.py` to:

   ```python
   import os
   OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
   ```

4. Set the variable in your environment, without committing its value:

   ```bash
   # macOS / Linux
   export OPENROUTER_API_KEY="YOUR_NEW_KEY"
   ```

   ```powershell
   # Windows PowerShell
   $env:OPENROUTER_API_KEY="YOUR_NEW_KEY"
   ```

### 5. Launch the app

```bash
streamlit run WoundDetection.py
```

Open the local URL displayed by Streamlit, upload an image, and inspect the segmentation and feature output. The chat interface requires a working OpenRouter connection and valid credentials.

## How the analysis works

1. **Preprocessing:** Images are converted to RGB at upload, then prepared for inference at 256 × 256 resolution with values scaled to `[0, 1]`.
2. **Segmentation:** The Keras model predicts a mask; predicted values greater than 0.5 are retained.
3. **Region extraction:** OpenCV finds contours, draws bounding boxes, and calculates geometry-based properties.
4. **Feature extraction:** Mean color, gray-level co-occurrence matrix (GLCM) texture statistics, and Canny edge responses characterize the image regions.
5. **Conversational interface:** The extracted feature summary is supplied as context to an OpenRouter-hosted LLM. The assistant displays responses within the Streamlit session.

**Measurement caveat:** Pixel area is not an area in cm² unless an image calibration method is added. The features also depend on image resolution, lighting, camera distance, and segmentation quality.

## Evaluation and reproducibility

The repository includes `eval_analysis.ipynb` and `wound-detection.ipynb` for analysis and model-related experimentation. This README intentionally does not report performance scores or dataset specifications that have not been independently checked against those notebooks. Consult them for training and evaluation details before publishing benchmark claims.

For reproducible results, future updates should document dataset sources and splits, preprocessing, model architecture, training settings, and test metrics (for example Dice and Intersection over Union).

## Known limitations

- The deployed checkpoint must be supplied at the path expected by the app.
- Segmentation and extracted features may be inaccurate on unfamiliar images or under different capture conditions.
- The severity index is an experimental mathematical heuristic, not a clinical risk score.
- LLM-generated explanations may be incorrect or incomplete.
- Uploading wound-related information to a third-party LLM service raises privacy considerations. Do not process identifiable patient data without appropriate authorization and safeguards.
- The current chat-state handling may preserve earlier wound context during subsequent uploads in the same session. Reset the session or update the implementation to bind each chat to the active image.

## Research reference

Related research publication:

*A Deep Learning Based Wound Segmentation and Characterization with LLM-Assisted Clinical Interpretability*  
[https://doi.org/10.1007/978-3-032-26028-4_4](https://doi.org/10.1007/978-3-032-26028-4_4)

## Author

**Fisha Mehabaw Alemayoh**  
[GitHub](https://github.com/fisha-mehabaw)

## License

No license is specified in the repository. Please add a `LICENSE` file if you intend to grant reuse permissions.
