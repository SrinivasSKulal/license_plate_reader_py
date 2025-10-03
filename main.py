from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
#conversion of image into an array
car_image = imread("car.jpg" , as_gray=True)
print(car_image)


# Getting the gray scale and binary version of the image
gray_car_image = car_image*255
fig , (ax1 , ax2) = plt.subplots(1,2)
ax1.imshow(gray_car_image , cmap="gray")
threshold_value = threshold_otsu(gray_car_image)
binary_car_image = gray_car_image > threshold_value
ax2.imshow(binary_car_image , cmap="gray")
plt.show()



# Using CCA(Connected Component Analysis) to group and label connected regions  on the foreground

from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import localization


#get the connected regions and connect them

label_image = measure.label(binary_car_image)
fig , (ax1) = plt.subplots(1)
ax1.imshow(gray_car_image , cmap="gray")



for region in regionprops(label_image):
    if region.area < 50:
        continue


    minRow , minCol , maxRow , maxCol = region.bbox

    rectBorder = patches.Rectangle((minCol , minRow) , maxRow-minCol , maxRow-minRow , edgecolor="red", linewidth=0.5 , fill=False)
    ax1.add_patch(rectBorder)


plt.show()


# Output shows regions other than the plate and hence we  hard code the width to 15-40 % of the image and height as  8-20% of the image

from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches

label_image = measure.label(binary_car_image)

plate_dimensions = (0.05*label_image.shape[0], 0.2*label_image.shape[0] , 0.05*label_image.shape[1] ,0.4*label_image.shape[1])
min_height  ,max_height, min_width , max_width  = plate_dimensions
plate_object_coordinates = []
plate_like_objects = []


fig , (ax1) = plt.subplots(1)
ax1.imshow(gray_car_image , cmap="gray")



for region in regionprops(label_image):
    if region.area < 50:
        continue


    minRow , minCol , maxRow , maxCol = region.bbox
    region_height = maxRow - minRow
    region_width = maxCol - minCol

    if region_height >= min_height and region_height <= max_height and region_width >= min_width  and region_width <= max_width and region_width > region_height:
        plate_like_objects.append(binary_car_image[minRow:maxRow , minCol : maxCol])
        plate_object_coordinates.append((minRow, minCol , maxRow , maxCol))

        rectBorder = patches.Rectangle((minCol , minRow) , maxRow-minCol , maxRow-minRow , edgecolor="red", linewidth=0.5 , fill=False)
        ax1.add_patch(rectBorder)


plt.show()

for i in plate_like_objects:
    print(i)
# Character segmentation

import numpy as  np
from skimage.transform import resize

license_plate = np.invert(plate_like_objects[2])
labelled_image = measure.label(license_plate)

fig ,ax1 = plt.subplots(1)
ax1.imshow(license_plate , cmap="gray")

character_dimensions = (0.35*license_plate.shape[0], 0.60*license_plate.shape[0], 0.05*license_plate.shape[1], 0.15*license_plate.shape[1])
min_height, max_height, min_width, max_width = character_dimensions

characters = []
counter = 0
column_list = []

for regions in regionprops(labelled_image):
    y0 , x0 , y1 , x1  = regions.bbox
    region_height = y1  - y0
    region_width = x1 - x0

    if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
        roi = license_plate[y0:y1 , x0:x1]
        rect_border = patches.Rectangle((x0,y1), x1 - x0  , y1 - y0 , edgecolor= "red")
        ax1.add_patch(rect_border)
        resized_char = resize(roi , (20,20))
        characters.append(resized_char)
        column_list.append(x0)
plt.show()
