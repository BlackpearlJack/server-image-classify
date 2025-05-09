import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d
from typing import List, Optional

# Globals
__class_name_to_number: dict = {}
__class_number_to_name: dict = {}
__model = None


def classify_image(image_base64_data: str, file_path: Optional[str] = None) -> List[dict]:
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)
    result = []

    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((
            scalled_raw_img.reshape(32 * 32 * 3, 1),
            scalled_img_har.reshape(32 * 32, 1)
        ))

        final = combined_img.reshape(1, -1).astype(float)
        prediction = __model.predict(final)[0]
        probabilities = np.round(__model.predict_proba(final) * 100, 2).tolist()[0]

        result.append({
            'class': class_number_to_name(prediction),
            'class_probability': probabilities,
            'class_dictionary': __class_name_to_number
        })

    return result


def class_number_to_name(class_num: int) -> str:
    return __class_number_to_name[class_num]


def load_saved_artifacts() -> None:
    global __class_name_to_number, __class_number_to_name, __model

    print("loading saved artifacts...start")

    with open("./artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    if __model is None:
        with open('./artifacts/saved_model.pkl', 'rb') as f:
            __model = joblib.load(f)

    print("loading saved artifacts...done")


def get_cv2_image_from_base64_string(b64str: str) -> np.ndarray:
    encoded_data = b64str.split(',')[1] if ',' in b64str else b64str
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def get_cropped_image_if_2_eyes(image_path: Optional[str], image_base64_data: Optional[str]) -> List[np.ndarray]:
    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    elif image_base64_data:
        img = get_cv2_image_from_base64_string(image_base64_data)
    else:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)

    return cropped_faces


def get_b64_test_image_for_lovato() -> str:
    with open("./b64.txt") as f:
        return f.read()


if __name__ == '__main__':
    load_saved_artifacts()
    # Example for testing
    # print(classify_image(get_b64_test_image_for_lovato()))
    # print(classify_image(None, "server/test-images/sample.jpg"))
