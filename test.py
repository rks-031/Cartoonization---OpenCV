import cv2
import numpy as np


def cartoonize_image(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply various edge detection operators

    # Prewitt's operator
    prewitt_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    prewitt_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    prewitt_mag = np.sqrt(prewitt_x**2 + prewitt_y**2)

    # Sobel's operator
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    # Kirsch operators
    kirsch1 = cv2.filter2D(
        gray, -1, np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]))
    kirsch2 = cv2.filter2D(
        gray, -1, np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]))
    kirsch3 = cv2.filter2D(
        gray, -1, np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]))

    # Canny operator
    canny_edges = cv2.Canny(gray, 100, 200)

    # LoG operator
    log_edges = cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_64F)

    # Combine all edge detection outputs
    combined_edges = np.maximum.reduce(
        [prewitt_mag, sobel_mag, kirsch1, kirsch2, kirsch3, canny_edges, np.abs(log_edges)])

    # Threshold the combined edges
    _, thresholded = cv2.threshold(combined_edges, 50, 255, cv2.THRESH_BINARY)

    # Convert thresholded image to uint8
    thresholded = np.uint8(thresholded)

    # Invert the thresholded image
    thresholded = cv2.bitwise_not(thresholded)

    # Create a mask for the cartoonization
    mask = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(img, mask)

    # Display the images
    cv2.imshow('Original', img)
    cv2.imshow('Edges', thresholded)
    cv2.imshow('Cartoonized', cartoon)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Call the function with the path to your image
cartoonize_image('images.png')
