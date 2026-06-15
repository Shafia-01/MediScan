import os
import io
import socket
import ipaddress
import logging
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image
import pymupdf

logger = logging.getLogger(__name__)


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def is_safe_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            logger.warning(f"SSRF Protection: Rejected URL '{url}' due to invalid scheme '{parsed.scheme}'. Only http and https are allowed.")
            return False

        hostname = parsed.hostname
        if not hostname:
            logger.warning(f"SSRF Protection: Rejected URL '{url}' due to empty hostname.")
            return False

        try:
            addr_info = socket.getaddrinfo(hostname, None)
        except socket.gaierror as e:
            logger.warning(f"SSRF Protection: Failed to resolve hostname '{hostname}': {e}")
            return False

        for info in addr_info:
            ip_str = info[4][0]
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                continue

            if ip.is_loopback or ip.is_link_local or ip.is_private:
                logger.warning(f"SSRF Protection: Rejected URL '{url}' because resolved IP '{ip_str}' is loopback, link-local, or private.")
                return False

            if ip_str == "169.254.169.254":
                logger.warning(f"SSRF Protection: Rejected URL '{url}' because resolved IP '{ip_str}' is a cloud metadata address.")
                return False

        return True
    except Exception as e:
        logger.error(f"SSRF Protection: Exception during safety check for '{url}': {e}")
        return False


def extract_images_from_url(url: str, output_folder: str = "extracted_images_url", timeout: int = 10, max_images: int = 50) -> list[str]:
    ensure_dir(output_folder)
    saved_paths: list[str] = []

    if not is_safe_url(url):
        logger.warning(f"Skipping extraction from unsafe URL: {url}")
        return saved_paths

    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            logger.error(f"Failed to retrieve the webpage: {response.status_code}")
            return saved_paths

        soup = BeautifulSoup(response.text, 'html.parser')
        images = soup.find_all('img')
        logger.info(f"Found {len(images)} images on the webpage")

        for idx, img in enumerate(images):
            if len(saved_paths) >= max_images:
                logger.info(f"Image extraction truncated at limit: {max_images}")
                break

            src = img.get('src')
            if not src:
                continue

            complete_url = urljoin(url, src)

            # SSRF check for image URLs
            if not is_safe_url(complete_url):
                logger.warning(f"Skipping unsafe image URL: {complete_url}")
                continue

            # Lightweight HEAD check and content-type filtering
            try:
                path_lower = complete_url.lower()
                if '.svg' in path_lower or '.avif' in path_lower:
                    logger.info(f"skipping unsupported type: {complete_url} (inferred from URL extension)")
                    continue

                content_type = ""
                try:
                    head_response = requests.head(complete_url, timeout=timeout, allow_redirects=True)
                    if head_response.status_code == 200:
                        content_type = head_response.headers.get('content-type', '').lower()
                except Exception as head_err:
                    logger.debug(f"HEAD request failed for {complete_url}: {head_err}")

                if content_type:
                    if 'svg' in content_type or 'avif' in content_type:
                        logger.info(f"skipping unsupported type: {complete_url} (content-type: {content_type})")
                        continue
                    if not content_type.startswith('image/'):
                        logger.info(f"Skipping non-image file: {complete_url} (content-type: {content_type})")
                        continue

                img_response = requests.get(complete_url, timeout=timeout)
                if img_response.status_code != 200:
                    continue

                # Final check in case HEAD wasn't used/available
                content_type = img_response.headers.get('content-type', '').lower()
                if not content_type.startswith('image/'):
                    logger.info(f"Skipping non-image file: {complete_url} (content-type: {content_type})")
                    continue

                if 'svg' in content_type or 'avif' in content_type:
                    logger.info(f"skipping unsupported type: {complete_url} (content-type: {content_type})")
                    continue

                if 'jpeg' in content_type or 'jpg' in content_type:
                    ext = 'jpg'
                elif 'png' in content_type:
                    ext = 'png'
                elif 'gif' in content_type:
                    ext = 'gif'
                elif 'webp' in content_type:
                    ext = 'webp'
                elif 'tiff' in content_type or 'tif' in content_type:
                    ext = 'tiff'
                else:
                    ext = 'jpg'

                filename = os.path.join(output_folder, f"url_image_{idx + 1}.{ext}")
                with open(filename, 'wb') as f:
                    f.write(img_response.content)

                try:
                    with Image.open(filename) as test_img:
                        test_img.verify()
                    saved_paths.append(filename)
                    logger.info(f"Downloaded: {filename}")
                except Exception as e:
                    logger.warning(f"Downloaded file is not a valid image: {filename} - {e}")
                    if os.path.exists(filename):
                        os.remove(filename)
            except Exception as e:
                logger.error(f"Failed to download {complete_url}: {e}")

        logger.info(f"Total valid images downloaded from URL: {len(saved_paths)}")
    except Exception as e:
        logger.error(f"Error processing URL {url}: {e}")

    return saved_paths


def extract_images_from_pdf(pdf_path: str, output_folder: str = "extracted_images_pdf", max_images: int = 50) -> list[str]:
    ensure_dir(output_folder)
    saved_paths: list[str] = []
    try:
        doc = pymupdf.open(pdf_path)
        len_xref = doc.xref_length()
        image_count = 0

        for xref in range(1, len_xref):
            if len(saved_paths) >= max_images:
                logger.info(f"Image extraction truncated at limit: {max_images}")
                break

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
                logger.info(f"Extracted: {filepath}")
            except Exception as e:
                logger.error(f"Failed to extract image {xref}: {e}")

        doc.close()
        logger.info(f"Total images extracted from PDF: {len(saved_paths)}")
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")

    return saved_paths
