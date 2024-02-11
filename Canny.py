import cv2
import numpy as np


def cartoonize_with_canny(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Convert edges to 3-channel for compatibility with bitwise_and
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Cartoonize the image
    cartoon = cv2.bitwise_and(img, edges)

    # Display the images
    cv2.imshow('Original', img)
    cv2.imshow('Edges (Canny)', edges)
    cv2.imshow('Cartoonized (Canny)', cartoon)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Call the function with the path to your image
cartoonize_with_canny('camera.jfif')
