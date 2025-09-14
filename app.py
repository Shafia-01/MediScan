import os
import tempfile
from typing import List, Dict
import pandas as pd
import streamlit as st
from PIL import Image

from image_classifier import ImageClassifier

THEME_COLORS = {
    "bg": "#d7eaef" ,
    "primary": "#011627",
    "success": "#07F068",
    "text": "#0C0C0C",
    "muted": "#12436C",
    "caption": "#011627",
    "tab_text": "#011627",
    "tab_selected": "#011627",
    "badge_medical": "#0A7D3A",
    "badge_non_med": "#E50808",
}

def apply_theme() -> None:
    css = f"""
    <style>
      .stApp {{
        background: {THEME_COLORS['bg']};
        color: {THEME_COLORS['text']};
      }}

      .stApp, .stApp * {{
        font-weight: 700 !important;
        font-family: Cambria, "Times New Roman", Times, serif !important;
      }}

      h1, h2, h3, h4, h5, h6 {{
        color: {THEME_COLORS['text']};
      }}

      .stButton > button {{
        background-color: {THEME_COLORS['primary']};
        color: #ffffff;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
      }}
      .stButton > button:hover {{
        background-color: #246da0;
        color: #ffffff;
      }}
      .stButton > button:focus {{
        outline: 2px solid {THEME_COLORS['success']};
      }}

      div[role="tablist"] {{
        display: flex !important;
        justify-content: center !important;
        gap: 1rem !important;
      }}
      div[role="tablist"] > button[role="tab"] {{
        border-bottom: 2px solid transparent !important;
        color: {THEME_COLORS.get('tab_text', THEME_COLORS['muted'])} !important;
        font-weight: 600 !important;
      }}
      div[role="tablist"] > button[role="tab"][aria-selected="true"] {{
        color: {THEME_COLORS.get('tab_selected', THEME_COLORS['primary'])} !important;
        border-bottom: 2px solid {THEME_COLORS.get('tab_selected', THEME_COLORS['primary'])} !important;
        font-weight: 700 !important;
      }}

      div[data-testid="stCaptionContainer"], .stCaption {{
        color: {THEME_COLORS.get('caption', THEME_COLORS['muted'])} !important;
        font-weight: 700 !important;
      }}
      .app-caption {{
        color: {THEME_COLORS.get('caption', THEME_COLORS['muted'])} !important;
        font-weight: 700 !important;
        font-size: 0.95rem;
        margin: 0.15rem 0 0.5rem 0;
      }}

      div[data-testid="stFileUploadDropzone"] {{
        background: #ffffff;
        border: 1.5px dashed {THEME_COLORS['primary']};
        border-radius: 10px;
      }}
      div[data-testid="stFileUploadDropzone"]:hover {{
        background: #eaf3fa;
      }}

      /* Center align common text */
      .stApp p, .stApp span, .stApp label, .stApp div, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
        text-align: center !important;
      }}

      .result-badge {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-weight: 600;
        color: #ffffff;
        font-size: 0.85rem;
        margin: 0 0 6px 2px;
      }}
      .badge-medical {{ background: {THEME_COLORS['badge_medical']}; }}
      .badge-nonmed {{ background: {THEME_COLORS['badge_non_med']}; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_classifier(model_path: str) -> ImageClassifier:
    return ImageClassifier(model_path=model_path)

def ensure_session_tempdir() -> str:
    if "_session_tmpdir" not in st.session_state:
        st.session_state._session_tmpdir = tempfile.mkdtemp(prefix="st_uploads_")
    return st.session_state._session_tmpdir

def save_upload_to_disk(upload, directory: str) -> str:
    file_path = os.path.join(directory, upload.name)
    with open(file_path, "wb") as f:
        f.write(upload.getbuffer())
    return file_path

def render_results(results: List[Dict]) -> None:
    if not results:
        st.info("No results to display.")
        return

    confident_results = [r for r in results if r.get("class") in {"medical", "non_medical"}]
    if not confident_results:
        st.info("No confident predictions to display.")
        return

    cols_per_row = 3
    for i in range(0, len(confident_results), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, result in zip(cols, confident_results[i : i + cols_per_row]):
            try:
                with Image.open(result["image_path"]) as pil_img:
                    pil_img = pil_img.convert("RGB")
                    badge_class = "badge-medical" if result["class"] == "medical" else "badge-nonmed"
                    col.markdown(
                        f"<span class='result-badge {badge_class}'>"
                        f"{result['class'].replace('_', ' ').title()}</span>",
                        unsafe_allow_html=True,
                    )
                    col.image(pil_img, use_container_width=True)
                    col.caption(
                        f"{os.path.basename(result['image_path'])} · Confidence: {result['confidence']:.1%}"
                    )
            except Exception as e:
                col.error(f"Failed to load image: {os.path.basename(result['image_path'])} ({e})")

    table_rows = [
        {
            "FILE": os.path.basename(r["image_path"]),
            "PREDICTION": r["class"],
            "CONFIDENCE": round(float(r["confidence"]) * 100.0, 2),
        }
        for r in confident_results
    ]
    st.write("Results table")
    df = pd.DataFrame(table_rows, columns=["FILE", "PREDICTION", "CONFIDENCE"]) 
    styled = (
        df.style
        .set_properties(**{
            "text-align": "center",
            "font-family": 'Cambria, "Times New Roman", Times, serif',
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("font-weight", "bold"),
                ("text-align", "center"),
                ("font-family", 'Cambria, "Times New Roman", Times, serif'),
            ]}
        ])
        .hide(axis="index")
    )
    st.write(styled)

    csv_lines = ["file,prediction,confidence_percent"] + [
        f"{row['FILE']},{row['PREDICTION']},{row['CONFIDENCE']}" for row in table_rows
    ]
    st.download_button(
        "Download CSV",
        data="\n".join(csv_lines).encode("utf-8"),
        file_name="classification_results.csv",
        mime="text/csv",
    )

def main() -> None:
    st.set_page_config(page_title="Med vs Non-med Image Classifier", layout="wide")
    apply_theme()
    st.title("Medical vs Non-medical Image Classifier")
    st.markdown(
        f"<div class='app-caption'>Upload images, a PDF, or provide a URL. The model will classify each image as medical or non-medical.</div>",
        unsafe_allow_html=True,
    )

    model_path = "image_classification_model.pth"
    threshold = 0.60
    use_tta = True

    classifier = load_classifier(model_path=model_path)

    tabs = st.tabs(["IMAGES", "PDF", "URL"])

    with tabs[0]:
        st.subheader(":blue[Upload image files]")
        uploads = st.file_uploader(
            "Select one or more images",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            accept_multiple_files=True,
        )
        if uploads:
            tmpdir = ensure_session_tempdir()
            image_paths: List[str] = []
            for up in uploads:
                try:
                    path = save_upload_to_disk(up, tmpdir)
                    image_paths.append(path)
                except Exception as e:
                    st.error(f"Failed to save upload: {up.name} ({e})")

            if st.button("Classify images", type="primary"):
                with st.spinner("Classifying..."):
                    results = classifier.classify_images(
                        image_paths,
                        confidence_threshold=threshold,
                        use_tta=use_tta,
                    )
                render_results(results)

    with tabs[1]:
        st.subheader(":blue[Upload a PDF to extract and classify images]")
        pdf_upload = st.file_uploader("PDF file", type=["pdf"], accept_multiple_files=False, key="pdf")
        if pdf_upload is not None:
            tmpdir = ensure_session_tempdir()
            pdf_path = save_upload_to_disk(pdf_upload, tmpdir)
            if st.button("Extract images and classify", type="primary"):
                with st.spinner("Extracting images from PDF..."):
                    extracted_paths = classifier.extract_images_from_pdf(pdf_path, output_folder=os.path.join(tmpdir, "pdf_images"))
                if not extracted_paths:
                    st.warning("No images were extracted from the PDF.")
                else:
                    with st.spinner("Classifying extracted images..."):
                        results = classifier.classify_images(
                            extracted_paths,
                            confidence_threshold=threshold,
                            use_tta=use_tta,
                        )
                    render_results(results)

    with tabs[2]:
        st.subheader(":blue[Provide a URL to scrape and classify images]")
        url = st.text_input("Enter a URL (http/https)")
        if st.button("Fetch images and classify", disabled=not bool(url.strip())):
            tmpdir = ensure_session_tempdir()
            with st.spinner("Downloading images from URL..."):
                extracted_paths = classifier.extract_images_from_url(url, output_folder=os.path.join(tmpdir, "url_images"))
            if not extracted_paths:
                st.warning("No valid images were found at the URL.")
            else:
                with st.spinner("Classifying downloaded images..."):
                    results = classifier.classify_images(
                        extracted_paths,
                        confidence_threshold=threshold,
                        use_tta=use_tta,
                    )
                render_results(results)

if __name__ == "__main__":
    main()


