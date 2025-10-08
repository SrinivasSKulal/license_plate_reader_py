import cv2
from matplotlib import pyplot as plt
import imutils
import numpy as np
import easyocr
import random

# --- Load image ---

file_name = "Untitled3.jpg"
img = cv2.imread(file_name)
if img is None:
    raise FileNotFoundError(f"Image not found. Make sure {file_name} exists in the same directory.")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")
plt.show()

# --- Convert to grayscale and smoothen ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

plt.imshow(cv2.cvtColor(bfilter, cv2.COLOR_BGR2RGB))
plt.title("Filtered Image")
plt.axis("off")
plt.show()

# --- Edge detection ---
edged = cv2.Canny(bfilter, 10, 300)  # Tune thresholds if needed
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
plt.title("Edge Detection Result")
plt.axis("off")
plt.show()

# --- Find contours ---
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]  # Top 30 largest contours

# --- Visualize all contours (for debugging) ---
img_contours = img.copy()
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
plt.title("All Detected Contours")
plt.axis("off")
plt.show()

# --- Find the contour with 4 corners (possible plate) ---
location = None
for contour in contours:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
    if len(approx) == 4:  # Looks like a rectangle
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 2 < aspect_ratio < 6:  # Typical license plate ratio
            location = approx
            break

print("Location:", location)

# --- If no plate found ---
if location is None:
    print("No license plate contour found. Try adjusting Canny or epsilon values.")
else:
    # --- Draw the detected plate contour ---
    mask = np.zeros(gray.shape, np.uint8)
    new_img = cv2.drawContours(mask, [location], 0, 255, -1)
    new_img = cv2.bitwise_and(img, img, mask=mask)

    plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
    plt.title("License Plate Region")
    plt.axis("off")
    plt.show()

    # --- Crop the plate area ---
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]

    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    plt.title("Cropped License Plate")
    plt.axis("off")
    plt.show()

    # --- Read text using EasyOCR ---
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)

    text = ""
    for detection in result:
        text += detection[1] + " "

    print("Detected License Plate Number:", text.strip())

    # --- Annotate image with detected plate ---
    annotated = img.copy()
    cv2.putText(annotated, text.strip(), (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.drawContours(annotated, [location], -1, (0, 255, 0), 3)

    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.title("Detected Plate with Text")
    plt.axis("off")
    plt.show()


