# NeuroVision AI - Streamlit web app

A shareable Streamlit Community Cloud app for the BRISC2025 four-class 2D brain MRI research classifier.

## Files required in the GitHub repository root

- `app.py`
- `requirements.txt`
- `BRISC2025_EfficientNetB0_finetuned.keras` (the trained model file)
- `.streamlit/config.toml`

## Important: add your model file

Copy `BRISC2025_EfficientNetB0_finetuned.keras` into this folder next to `app.py` before uploading the project to GitHub. The model file is not included in this starter package.

## Run locally

Use a Python environment compatible with TensorFlow 2.21 (Python 3.13 is supported):

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy

1. Create a GitHub repository. If you want anyone to access the app, make the repository public.
2. Upload all four required items listed above, preserving the `.streamlit/config.toml` folder structure.
3. Sign in to Streamlit Community Cloud using GitHub.
4. Create an app and select the repository, branch `main`, and entrypoint `app.py`.
5. In Advanced settings, select Python 3.13, then deploy.
6. After a successful deployment, use the assigned `*.streamlit.app` URL to share the web app.

## Privacy and research limitations

A public repository exposes the model file to anyone who can access the repository. Do not upload private patient data, identifiable MRI scans, credentials, or secret files. The app processes uploaded images during a session and does not intentionally persist them in this code. This is a research prototype, not a medical device; scores are not clinical diagnoses or calibrated probabilities.
