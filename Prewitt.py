import cv2
import numpy as np


def cartoonize_with_prewitt(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Prewitt's operator
    prewitt_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    prewitt_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    prewitt_mag = np.sqrt(prewitt_x**2 + prewitt_y**2)

    # Threshold the edges
    _, edges = cv2.threshold(prewitt_mag, 50, 255, cv2.THRESH_BINARY)

    # Convert edges to uint8
    edges = np.uint8(edges)

    # Invert the edges
    edges = cv2.bitwise_not(edges)

    # Convert edges to 3-channel for compatibility with bitwise_and
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Cartoonize the image
    cartoon = cv2.bitwise_and(img, edges)

    # Display the images
    cv2.imshow('Original', img)
    cv2.imshow('Edges (Prewitt)', edges)
    cv2.imshow('Cartoonized (Prewitt)', cartoon)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Call the function with the path to your image
cartoonize_with_prewitt('camera.jfif')
