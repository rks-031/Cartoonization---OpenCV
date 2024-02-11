import cv2


def cartoonize_with_log(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (3, 3), 0)

    # Apply Laplacian edge detection
    edges = cv2.Laplacian(blurred, cv2.CV_64F)
    edges = cv2.convertScaleAbs(edges)

    # Threshold the edges
    _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

    # Invert the edges
    edges = cv2.bitwise_not(edges)

    # Cartoonize the image
    cartoon = cv2.bitwise_and(img, img, mask=edges)

    # Display the images
    cv2.imshow('Original', img)
    cv2.imshow('Edges (LoG)', edges)
    cv2.imshow('Cartoonized (LoG)', cartoon)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Call the function with the path to your image
cartoonize_with_log('camera.jfif')
