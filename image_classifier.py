import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageFile
import warnings
from image_extractors import extract_images_from_pdf as ext_from_pdf, extract_images_from_url as ext_from_url
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import shutil

warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes",
    category=UserWarning,
    module="PIL.Image",
)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageClassifier:
    def __init__(self, model_path='image_classification_model.pth'):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_names = ['medical', 'non_medical']
        self.num_classes = len(self.class_names)
        
        if os.path.exists(model_path):
            self.model = models.resnet18(weights=None)
        else:
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)
        
        class_map_path = 'class_to_idx.json'
        if os.path.exists(class_map_path):
            try:
                with open(class_map_path, 'r') as f:
                    class_to_idx = json.load(f)
                idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
                self.class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
                self.num_classes = len(self.class_names)
                self.model.fc = nn.Linear(num_ftrs, self.num_classes)
                print(f"Loaded class mapping: {self.class_names}")
            except Exception as map_err:
                print(f"Warning: Failed to load class_to_idx.json ({map_err}). Using default class order {self.class_names}.")

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found. Using untrained model.")
        
        self.model.eval()
        self.model.to(self.device)
        
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _predict_tensor(self, input_batch: torch.Tensor) -> tuple[str, float, np.ndarray]:
        with torch.no_grad():
            outputs = self.model(input_batch)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probabilities))
        pred_class = self.class_names[pred_idx]
        confidence = float(probabilities[pred_idx])
        return pred_class, confidence, probabilities

    def classify_image(self, image_path, confidence_threshold: float = 0.60, use_tta: bool = True):
        try:
            if not os.path.exists(image_path):
                print(f"File does not exist: {image_path}")
                return None

            file_size_bytes = os.path.getsize(image_path)
            if file_size_bytes < 100:
                print(f"File too small, likely corrupted: {image_path} ({file_size_bytes} bytes)")
                return None

            try:
                with Image.open(image_path) as test_img:
                    test_img.verify()
            except Exception as integrity_error:
                print(f"Invalid or corrupted image file: {image_path} - {integrity_error}")
                return None

            image = Image.open(image_path).convert('RGB')

            base_tensor = self.preprocess(image)
            probs_accumulator = None

            if use_tta:
                augmentations = [
                    base_tensor,
                    self.preprocess(Image.fromarray(np.array(image)[:, ::-1, :]))
                ]
                for t in augmentations:
                    batch = t.unsqueeze(0).to(self.device)
                    _, _, probs = self._predict_tensor(batch)
                    probs_accumulator = probs if probs_accumulator is None else (probs_accumulator + probs)
                probs_mean = probs_accumulator / len(augmentations)
                pred_idx = int(np.argmax(probs_mean))
                predicted_class_name = self.class_names[pred_idx]
                confidence = float(probs_mean[pred_idx])
                probabilities = probs_mean
            else:
                input_batch = base_tensor.unsqueeze(0).to(self.device)
                predicted_class_name, confidence, probabilities = self._predict_tensor(input_batch)

            final_class = predicted_class_name if confidence >= confidence_threshold else 'uncertain'

            return {
                'class': final_class,
                'confidence': confidence,
                'probabilities': probabilities,
                'image_path': image_path
            }

        except Exception as e:
            print(f"Error classifying {image_path}: {e}")
            return None
    
    def extract_images_from_pdf(self, pdf_path, output_folder="extracted_images_pdf"):
        return ext_from_pdf(pdf_path, output_folder=output_folder)
    
    def extract_images_from_url(self, url, output_folder="extracted_images_url"):
        return ext_from_url(url, output_folder=output_folder)
    
    def classify_images(self, image_paths, confidence_threshold: float = 0.60, use_tta: bool = True,
                        save_uncertain: bool = False, uncertain_dir: str = "uncertain_images"):
        results = []
        if save_uncertain and not os.path.exists(uncertain_dir):
            os.makedirs(uncertain_dir, exist_ok=True)

        for path in image_paths:
            result = self.classify_image(path, confidence_threshold=confidence_threshold, use_tta=use_tta)
            if result is not None:
                if save_uncertain and result['class'] == 'uncertain':
                    try:
                        dest = os.path.join(uncertain_dir, os.path.basename(path))
                        shutil.copy2(path, dest)
                    except Exception as copy_err:
                        print(f"Warning: failed to copy uncertain image {path} -> {uncertain_dir}: {copy_err}")
                results.append(result)
        return results
    
    def display_results(self, results):
        if not results:
            print("No results to display")
            return
    
        num_images = len(results)
        cols = min(3, num_images)
        rows = (num_images + cols - 1) // cols
    
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1 and cols == 1:
            axes_grid = np.array([[axes]])
        elif rows == 1:
            axes_grid = np.array([axes])
        elif cols == 1:
            axes_grid = axes.reshape(rows, 1)
        else:
            axes_grid = axes
            
        for idx, result in enumerate(results):
            row = idx // cols
            col = idx % cols
            
            ax = axes_grid[row][col]
            try:
                image = Image.open(result['image_path'])
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                img_array = np.array(image)
            
                if len(img_array.shape) == 3 and img_array.shape[2] in [1, 3, 4]:
                    if img_array.shape[2] == 1:
                        img_array = img_array.squeeze()
                    elif img_array.shape[2] == 4:
                        img_array = img_array[:, :, :3]
                
                    ax.imshow(img_array)
                    ax.axis('off')
                
                    title = f"Predicted: {result['class']}\nConfidence: {result['confidence']:.2%}"
                    if result['class'] == 'medical':
                        bg_color = 'red'
                    elif result['class'] == 'non_medical':
                        bg_color = 'blue'
                    else: 
                        bg_color = 'orange'
                    ax.set_title(title, fontsize=10, color='white', backgroundcolor=bg_color)
                else:
                    ax.text(0.5, 0.5, f"Unsupported\nImage Format\n{result['class']}\n{result['confidence']:.2%}", 
                            ha='center', va='center', transform=ax.transAxes,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='red' if result['class'] == 'medical' else 'blue'))
                    ax.axis('off')
                    print(f"Unsupported image format for {result['image_path']}: shape {img_array.shape}")
                
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading image\n{result['class']}\n{result['confidence']:.2%}\nError: {str(e)[:20]}", 
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='gray'))
                ax.axis('off')
                print(f"Error displaying image {result['image_path']}: {e}")
    
        for idx in range(num_images, rows * cols):
            row = idx // cols
            col = idx % cols
            axes_grid[row][col].axis('off')
    
        plt.tight_layout()
        plt.show()
    
    def print_summary(self, results):
        if not results:
            print("No results to summarize")
            return
        
        medical_count = sum(1 for r in results if r['class'] == 'medical')
        non_medical_count = sum(1 for r in results if r['class'] == 'non_medical')
        uncertain_count = sum(1 for r in results if r['class'] == 'uncertain')
        
        print("\n" + "="*50)
        print("CLASSIFICATION SUMMARY")
        print("="*50)
        total = len(results)
        print(f"Total images processed: {total}")
        print(f"Medical images: {medical_count} ({(medical_count/total*100 if total else 0):.1f}%)")
        print(f"Non-medical images: {non_medical_count} ({(non_medical_count/total*100 if total else 0):.1f}%)")
        print(f"Uncertain images: {uncertain_count} ({(uncertain_count/total*100 if total else 0):.1f}%)")
        print("\nDetailed Results:")
        print("-"*50)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {os.path.basename(result['image_path'])}: {result['class']} (Confidence: {result['confidence']:.2%})")

def main():
    parser = argparse.ArgumentParser(description='Classify images from PDF or URL')
    parser.add_argument('--input', required=True, help='Input: PDF file path or URL')
    parser.add_argument('--model', default='image_classification_model.pth', help='Path to trained model')
    parser.add_argument('--display', action='store_true', help='Display results with images')
    parser.add_argument('--save-results', action='store_true', help='Save results to file')
    parser.add_argument('--uncertain-threshold', type=float, default=0.60, help='Confidence threshold below which predictions are labeled uncertain')
    parser.add_argument('--no-tta', action='store_true', help='Disable test-time augmentation for faster but possibly less robust predictions')
    parser.add_argument('--save-uncertain', action='store_true', help='Copy uncertain images to a folder for manual review')
    parser.add_argument('--uncertain-dir', type=str, default='uncertain_images', help='Directory to store uncertain images when --save-uncertain is used')
    
    args = parser.parse_args()
    classifier = ImageClassifier(args.model)
    input_path = args.input
    
    if input_path.lower().endswith('.pdf'):
        print(f"Processing PDF: {input_path}")
        extracted_images = classifier.extract_images_from_pdf(input_path)
    elif input_path.startswith(('http://', 'https://')):
        print(f"Processing URL: {input_path}")
        extracted_images = classifier.extract_images_from_url(input_path)
    else:
        print("Invalid input. Please provide a PDF file path or a valid URL.")
        return
    
    if not extracted_images:
        print("No images were extracted from the input.")
        return
    
    print(f"\nClassifying {len(extracted_images)} images...")
    results = classifier.classify_images(
        extracted_images,
        confidence_threshold=args.uncertain_threshold,
        use_tta=not args.no_tta,
        save_uncertain=args.save_uncertain,
        uncertain_dir=args.uncertain_dir
    )
    
    classifier.print_summary(results)
    
    if args.display:
        classifier.display_results(results)
    
    if args.save_results:
        output_file = "classification_results.txt"
        with open(output_file, 'w') as f:
            f.write("Image Classification Results\n")
            f.write("="*30 + "\n\n")
            for result in results:
                f.write(f"Image: {os.path.basename(result['image_path'])}\n")
                f.write(f"Predicted Class: {result['class']}\n")
                f.write(f"Confidence: {result['confidence']:.2%}\n")
                f.write("-"*20 + "\n")
        print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print("Use the Streamlit app for an interactive UI: ")
        print("python -m streamlit run app.py")
        print("")
        print("For CLI usage, run with --help: ")
        print("python image_classifier.py --help")
    else:
        main()
