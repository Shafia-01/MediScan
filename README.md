# Medical vs Non‑Medical Image Classifier

An end‑to‑end system to classify images as medical vs non‑medical. It includes a Streamlit app for uploading images/PDFs or fetching images from a URL, and a PyTorch ResNet18 model trained on a small dataset.

## Quickstart

```bash
pip install -r requirements.txt

# (Optional) Train or retrain
python training_model.py

# Launch UI
python -m streamlit run app.py
```

The app supports:
- Images tab: upload one or more images
- PDF tab: upload a PDF (images are extracted and classified)
- URL tab: fetch images from a web page and classify

## My approach and reasoning

- Baseline backbone: ResNet18 with ImageNet normalization. It’s a strong, lightweight baseline for binary image classification with good latency on CPU and GPU.
- Transfer learning: Replace the final FC layer to output 2 classes (medical, non_medical). Start with ImageNet weights for better feature reuse; fine‑tune with a staged schedule.
- Data pipeline: Standard transforms (Resize 256 → CenterCrop 224 for val; stronger augmentation for train: RandomResizedCrop, flips, rotations, color jitter) to improve generalization.
- Class balance: Use a `WeightedRandomSampler` based on observed class counts to mitigate class imbalance.
- Optimization: AdamW + StepLR scheduler with warm‑up/fine‑tuning. First epochs train only the FC layer; later unfreeze all layers to refine features.
- Inference robustness: Optional TTA (horizontal flip) and an uncertainty threshold; predictions below threshold are labeled as “uncertain.” The Streamlit UI hides uncertain results per the product requirement.
- UX: Streamlit front‑end with images/PDF/URL flows, grid previews, and CSV export. No settings sidebar; sensible defaults are hardcoded.

## Accuracy results on a small validation set

Evaluated on `data/val` using `evaluate.py` with threshold 0.60 and TTA disabled:

```text
Overall accuracy: 100.0% (101/101)
Per‑class accuracy: medical 100.0%, non_medical 100.0%
Uncertain predictions: 0
Avg inference time: ~52.09 ms/image on CPU (ResNet18)
```

Reproduce locally:

```bash
python evaluate.py
```

Notes:
- This is a small validation set; real‑world accuracy will vary with data distribution and image quality. Consider a larger, stratified test set and k‑fold validation for stronger estimates.

## Performance and efficiency considerations

- Model size/latency: ResNet18 offers a good trade‑off between accuracy and speed. On CPU, ~50–60 ms/image typical for 224×224; GPUs are faster.
- Preprocessing: PIL + torchvision transforms keep overhead low; center crop at inference; batch size 1 in UI to simplify memory use.
- TTA: Disabled by default in evaluation; in UI it’s enabled but can be toggled in code. TTA increases latency ~2× with modest robustness gains.
- Device selection: Auto‑selects CUDA if available, otherwise CPU. All tensor ops pinned to selected device.
- Memory: Single‑image inference path keeps memory footprint small; no large batches in the UI flow.
- I/O: URL/PDF extractors verify image integrity and skip invalid/tiny files to avoid wasting compute.

## Project structure

```
project/
├── app.py                       # Streamlit frontend (images/PDF/URL)
├── image_classifier.py          # Classifier class and CLI entry
├── image_extractors.py          # PDF/URL image extraction utilities
├── training_model.py            # Training script (transfer learning)
├── evaluate.py                  # Small validation evaluator (data/val)
├── image_classification_model.pth  # Trained weights (after training)
├── class_to_idx.json            # Class mapping saved at train time
├── requirements.txt             # Python dependencies
└── data/
    ├── train/
    │   ├── medical/
    │   └── non_medical/
    └── val/
        ├── medical/
        └── non_medical/
```

## Troubleshooting

- “Model file not found”: Train first (`python training_model.py`) or place `image_classification_model.pth` in the project root.
- PDF or URL extraction issues: Ensure `pymupdf`, `requests`, and `beautifulsoup4` are installed and the source contains extractable images.
- CUDA issues: The code falls back to CPU automatically.

## License

This project is for educational and research purposes.

