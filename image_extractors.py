import os
import io
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from PIL import Image
import pymupdf


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def extract_images_from_url(url: str, output_folder: str = "extracted_images_url", timeout: int = 10) -> list[str]:
    ensure_dir(output_folder)
    saved_paths: list[str] = []
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            print(f"Failed to retrieve the webpage: {response.status_code}")
            return saved_paths

        soup = BeautifulSoup(response.text, 'html.parser')
        images = soup.find_all('img')
        print(f"Found {len(images)} images on the webpage")

        for idx, img in enumerate(images):
            src = img.get('src')
            if not src:
                continue

            complete_url = urljoin(url, src)
            try:
                img_response = requests.get(complete_url, timeout=timeout)
                if img_response.status_code != 200:
                    continue

                content_type = img_response.headers.get('content-type', '').lower()
                if not content_type.startswith('image/'):
                    print(f"Skipping non-image file: {complete_url} (content-type: {content_type})")
                    continue

                if 'jpeg' in content_type or 'jpg' in content_type:
                    ext = 'jpg'
                elif 'png' in content_type:
                    ext = 'png'
                elif 'gif' in content_type:
                    ext = 'gif'
                elif 'webp' in content_type:
                    ext = 'webp'
                else:
                    ext = 'jpg'

                filename = os.path.join(output_folder, f"url_image_{idx + 1}.{ext}")
                with open(filename, 'wb') as f:
                    f.write(img_response.content)

                try:
                    with Image.open(filename) as test_img:
                        test_img.verify()
                    saved_paths.append(filename)
                    print(f"Downloaded: {filename}")
                except Exception as e:
                    print(f"Downloaded file is not a valid image: {filename} - {e}")
                    if os.path.exists(filename):
                        os.remove(filename)
            except Exception as e:
                print(f"Failed to download {complete_url}: {e}")

        print(f"Total valid images downloaded from URL: {len(saved_paths)}")
    except Exception as e:
        print(f"Error processing URL {url}: {e}")

    return saved_paths


def extract_images_from_pdf(pdf_path: str, output_folder: str = "extracted_images_pdf") -> list[str]:
    ensure_dir(output_folder)
    saved_paths: list[str] = []
    try:
        doc = pymupdf.open(pdf_path)
        len_xref = doc.xref_length()
        image_count = 0

        for xref in range(1, len_xref):
            try:
                subtype = doc.xref_get_key(xref, "Subtype")[1]
            except Exception:
                continue
            if subtype != "/Image":
                continue

            try:
                imgdata = doc.extract_image(xref)
                pil_image = Image.open(io.BytesIO(imgdata['image']))
                image_count += 1
                img_format = imgdata.get('ext', 'png')
                filename = f"pdf_image_{image_count}.{img_format}"
                filepath = os.path.join(output_folder, filename)
                pil_image.save(filepath)
                saved_paths.append(filepath)
                print(f"Extracted: {filepath}")
            except Exception as e:
                print(f"Failed to extract image {xref}: {e}")

        doc.close()
        print(f"Total images extracted from PDF: {len(saved_paths)}")
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")

    return saved_paths


