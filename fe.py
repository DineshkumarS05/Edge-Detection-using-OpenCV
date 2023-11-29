import cv2
import numpy as np
import streamlit as st

def edge_detection_canny(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

def edge_detection_log(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Laplacian(blurred, cv2.CV_64F)
    edges = np.uint8(np.absolute(edges))
    return edges

def edge_detection_dog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred1 = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred2 = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = blurred1 - blurred2
    return edges

def line_detection_hough(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    result = image.copy()
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return result

def corner_detection_harris(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)
    image[corners > 0.01 * corners.max()] = [0, 0, 255]
    return image

def corner_detection_hessian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hessian = cv2.xfeatures2d.StarDetector_create()
    keypoints = hessian.detect(gray)
    result = cv2.drawKeypoints(image, keypoints, None, color=(0, 0, 255))
    return result



# Streamlit UI
st.title("Image Feature Extraction")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

    option = st.selectbox("Choose feature extraction method:", 
                        ["Canny Edge", "LOG Edge", "DOG Edge", 
                        "Hough Lines", "Harris Corner", "Hessian Corner"])


    st.image(image, caption="Uploaded Image", use_column_width=True)



    if option == "Canny Edge":
        result = edge_detection_canny(image)
    elif option == "LOG Edge":
        result = edge_detection_log(image)
    elif option == "DOG Edge":
        result = edge_detection_dog(image)
    elif option == "Hough Lines":
        result = line_detection_hough(image)
    elif option == "Harris Corner":
        result = corner_detection_harris(image)
    elif option == "Hessian Corner":
        result = corner_detection_hessian(image)

    else:
        result = None  # Handle the case when option is not recognized

    if result is not None:
        st.image(result, caption="Result", use_column_width=True)
    else:
        st.write("Selected option not implemented or produced no result.")




