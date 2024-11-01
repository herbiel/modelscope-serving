import base64

import cv2
import numpy as np
import requests

def numpy_to_base64(image_np, image_ext):
    data = cv2.imencode('.' + image_ext, image_np)[1]
    image_bytes = data.tobytes()
    image_base4 = base64.b64encode(image_bytes).decode('utf8')
    return image_base4


def numpy_to_bytes(image_np, image_ext):
    data = cv2.imencode('.' + image_ext, image_np)[1]
    image_bytes = data.tobytes()
    return image_bytes


def numpy_to_file(image_np, filename):
    cv2.imwrite(filename, image_np)
    return filename


def bytes_to_numpy(image_bytes):
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image_np2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image_np2


def bytes_to_base64(image_bytes):
    image_base4 = base64.b64encode(image_bytes).decode('utf8')
    return image_base4


def bytes_to_file(image_bytes, filename):
    with open(filename, 'wb') as f:
        f.write(image_bytes)
        return filename


def file_to_numpy(path_file):
    image_np = cv2.imread(path_file)
    return image_np


def file_to_bytes(path_file):
    with open(path_file, 'rb') as f:
        image_bytes = f.read()
        return image_bytes


def file_to_base64(path_file):
    with open(path_file, 'rb') as f:
        image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf8')
        return image_base64


def base64_to_bytes(image_base64):
    image_bytes = base64.b64decode(image_base64)
    return image_bytes


def base64_to_numpy(image_base64):
    image_bytes = base64.b64decode(image_base64)
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image_np2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image_np2


def base64_to_file(image_base64, filename):
    image_bytes = base64.b64decode(image_base64)
    with open(filename, 'wb') as f:
        f.write(image_bytes)
        return filename


def url_to_numpy(image_url):
    try:
        # Fetch the image from the URL
        response = requests.get(image_url)

        # Check if the request was successful
        if response.status_code != 200:
            raise ValueError("Failed to fetch the image from the URL")

        # Convert the image content to a NumPy array
        image_np = np.frombuffer(response.content, dtype=np.uint8)

        # Decode the NumPy array into an image
        image_np2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Check if decoding was successful
        if image_np2 is None:
            raise ValueError("Image decoding failed.")

        return image_np2

    except Exception as e:
        print(f"Error in url_to_numpy: {e}")
        return None