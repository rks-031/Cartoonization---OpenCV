import cv2
import numpy as np


def cartoonize_with_kirsch(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Kirsch operator kernels
    kirsch1 = np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]])
    kirsch2 = np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]])
    kirsch3 = np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]])
    kirsch4 = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]])
    kirsch5 = np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])
    kirsch6 = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])
    kirsch7 = np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
    kirsch8 = np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]])
    kernels = [kirsch1, kirsch2, kirsch3, kirsch4,
               kirsch5, kirsch6, kirsch7, kirsch8]

    # Apply Kirsch operators
    kirsch_responses = [cv2.filter2D(gray, -1, kernel) for kernel in kernels]

    # Combine Kirsch responses
    combined_edges = np.maximum.reduce(kirsch_responses)

    # Threshold the edges
    _, edges = cv2.threshold(combined_edges, 50, 255, cv2.THRESH_BINARY)

    # Convert edges to 3-channel for compatibility with bitwise_and
    edges = cv2.cvtColor(np.uint8(edges), cv2.COLOR_GRAY2BGR)

    # Cartoonize the image
    cartoon = cv2.bitwise_and(img, edges)

    # Create a named window with a small size
    cv2.namedWindow('Cartoonized (Kirsch)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Cartoonized (Kirsch)', 400, 300)

    # Display the images
    cv2.imshow('Original', img)
    cv2.imshow('Edges (Kirsch)', edges)
    cv2.imshow('Cartoonized (Kirsch)', cartoon)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Call the function with the path to your image
cartoonize_with_kirsch('camera.jfif')
