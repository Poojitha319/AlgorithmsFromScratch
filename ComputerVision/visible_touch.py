import cv2
import numpy as np

# Callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Update touch coordinates based on mouse position
        touch_coordinates[0], touch_coordinates[1] = x, y
        update_visible_touch()

# Function to apply visible touch effect
def apply_visible_touch(img, touch_coordinates, touch_radius=20, color=(0, 255, 0), thickness=-1):
    # Create a copy of the input image to avoid modifying the original
    img_with_touch = img.copy()

    # Draw a filled circle at the touch coordinates
    cv2.circle(img_with_touch, touch_coordinates, touch_radius, color, thickness)

    return img_with_touch

# Function to update visible touch based on dynamic input
def update_visible_touch():
    global original_image, touch_coordinates
    image_with_touch = apply_visible_touch(original_image, touch_coordinates)
    cv2.imshow("Image with Visible Touch", image_with_touch)

# Read an image from file
image_path = r"C:\Users\Sai\Pictures\grp.jpg"
original_image = cv2.imread(image_path)

# Check if the image is successfully loaded
if original_image is None:
    print("Error: Unable to load the image.")
else:
    # Initialize touch coordinates
    touch_coordinates = [original_image.shape[1] // 2, original_image.shape[0] // 2]

    # Create a window and set the mouse callback function
    cv2.namedWindow("Image with Visible Touch")
    cv2.setMouseCallback("Image with Visible Touch", mouse_callback)

    while True:
        # Display the original image
        cv2.imshow("Original Image", original_image)

        # Wait for a key event (0 milliseconds means wait indefinitely)
        key = cv2.waitKey(0)

        # Exit the loop if the 'Esc' key is pressed
        if key == 27:
            break

    cv2.destroyAllWindows()
