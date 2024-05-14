#!pip install ultralytics
import os
import csv
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from skimage.morphology import skeletonize
#folder delete
import shutil
import pandas as pd
from tqdm import tqdm



%cd /content
##to find the pixel and feet
img1 = cv2.imread("/content/redline1.jpg")

lowcolor1 = (0, 0, 200)
highcolor1 = (50, 50, 255)
thresh = cv2.inRange(img1, lowcolor1, highcolor1)

# Apply morphology close
kernel = np.ones((5, 5), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Get contours and filter on area
result = img1.copy()
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
result = img1.copy()
for c in contours:
    area = cv2.contourArea(c)
    if area > 5000:
        cv2.drawContours(result, [c], -1, (0, 255, 0), 2)

# Save resulting images
cv2.imwrite('red_line_thresh11.png', thresh)

# Load the binary mask image
mask = cv2.imread('red_line_thresh11.png', cv2.IMREAD_GRAYSCALE)

# Define the color value for white
white_color = 255

# Find the coordinates of all white pixels in the image
white_pixels = np.column_stack(np.where(mask == white_color))

# Check if there are any white pixels in the image
if len(white_pixels) > 0:
    # Find the leftmost and rightmost coordinates
    leftmost_coordinate = tuple(white_pixels[white_pixels[:, 1].argmin()])
    rightmost_coordinate = tuple(white_pixels[white_pixels[:, 1].argmax()])

    # Calculate the length in pixels
    length_in_pixels = rightmost_coordinate[1] - leftmost_coordinate[1]




# Open the text file for reading
with open("IMG15_len.txt", "r") as file:
    # Read the single line from the file
    line = file.readline()

# Extract the numeric part from the line
numeric_value = ''.join(char for char in line if char.isdigit() or char == '.')

# Check if a numeric value was found
if numeric_value:
    numeric_value = float(numeric_value)  # Convert to float if you want to work with numbers


#getting the feet by pixel
length= numeric_value/length_in_pixels
print(length)
# above code we have attained the feet/pixel

# Define the path for the CSV file and the folder for chipped images
HOME= os.getcwd()
csv_file_path = os.path.join(HOME, 'object_detection_results.csv')
chipped_image_folder = os.path.join(HOME, 'Chipped_image')

# Define the path for the text file
txt_file_path = os.path.join(HOME, 'Log detected.txt')
# Define the path for the number of wood logs whose circumference is less than 12 inches
txt_path_logSec1 = os.path.join(HOME, 'Log_less_than_12.txt')
# Define the path for the number of wood logs whose circumference is greater than equal 12 inches
#and less than equal 38 inches
txt_path_logSec2 = os.path.join(HOME, 'Log_from_12_to_38.txt')
#Define the path for the number of wood logs whose circumference is greater than 38 inches
txt_path_logSec3 = os.path.join(HOME, 'Log_greater_than_38.txt')


# Ensuring the chipped_images folder exists
os.makedirs(chipped_image_folder, exist_ok=True)

# Load the YOLO model
model = YOLO(f'/content/detect.pt')

# Perform object detection
results = model.predict(source='/content/redline1.jpg', conf=0.25, save=True)
cnt = 0
lessThan12=0
from12To38=0
greaterThan38=0

# Create and open the CSV file for writing
with open(csv_file_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    for indexx, numx in enumerate(results):
        num_woods = len(results[indexx].boxes.xyxy)

        # Write the column headers
        csvwriter.writerow(["Index", "left top", "bottom right", "Horizontal diameter", "Vertical diameter", "Diameter_pixel","Diameter_feet","Diameter_inches","Circumference_feet","Circumference_inches"])

        for indexy, numy in enumerate(results[indexx].boxes.xyxy):
            box = results[indexx].boxes.xyxy[indexy]

            # Extract coordinates and calculate dimensions
            left_top = (int(box[0]), int(box[1]))
            bottom_right = (int(box[2]), int(box[3]))
            horizontal_diameter = float(box[2] - box[0])
            vertical_diameter = float(box[3] - box[1])
            diameter_pixel = (horizontal_diameter + vertical_diameter) / 2
            diameter_feet=diameter_pixel* length
            diameter_inches=diameter_feet*12
            circumference_feet= 2*3.14*(diameter_feet/2)
            circumference_inches=2*3.14*(diameter_inches/2)

            # Increment the count
            cnt += 1

            # Writing the data to the CSV file
            csvwriter.writerow([cnt, left_top, bottom_right, horizontal_diameter, vertical_diameter, diameter_pixel, diameter_feet,diameter_inches,circumference_feet,circumference_inches])

            # Crop and save the chipped image
            image = Image.open('/content/redline1.jpg')
            chipped_image = image.crop((left_top[0]-10, left_top[1]-10, bottom_right[0]+10, bottom_right[1]+10))

            # Convert the image to 'RGB' mode before saving as JPEG
            chipped_image = chipped_image.convert('RGB')

            chipped_image.save(os.path.join(chipped_image_folder, f"{cnt}.jpg"))
            if circumference_inches<12:
              lessThan12=lessThan12+1
            elif  circumference_inches>=12 and circumference_inches<=38:
              from12To38=from12To38+1
            else:
              greaterThan38=greaterThan38+1



# Write the value of cnt to the text file
with open(txt_file_path, 'w') as txtfile:
    txtfile.write(str(cnt))
#Write the values of circumference_cnt of 3 diff section to text file
with open(txt_path_logSec1 , 'w') as txtfile:
    txtfile.write(str(lessThan12))
with open(txt_path_logSec2 , 'w') as txtfile:
    txtfile.write(str(from12To38))
with open(txt_path_logSec3 , 'w') as txtfile:
    txtfile.write(str(greaterThan38))


#Standardization of all the images to 100x100 pixels
target_size = (100, 100)

# Path to the augmented images
augmented_images_dir = "/content/Chipped_image"

# Path to save standardized images
standardized_images_dir = "/content/Chipped_image"

if not os.path.exists(standardized_images_dir):
    os.makedirs(standardized_images_dir)

# Iterate through each augmented image
for image_name in tqdm(os.listdir(augmented_images_dir)):
    image_path = os.path.join(augmented_images_dir, image_name)

    # Read the image
    img = cv2.imread(image_path)

    # Resize the image to the target size with INTER_CUBIC interpolation
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)

    # Save the standardized image
    standardized_image_path = os.path.join(standardized_images_dir, image_name)
    cv2.imwrite(standardized_image_path, img_resized)




'''
-------------------------------------------------------------------------------
#log roundness
-------------------------------------------------------------------------------
'''


#PART1-> cropped image to the binary masked  image

def segment_foreground(image_path):
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to improve binary mask quality
    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove small noise using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if contours are found
    if not contours:
        #unable  to do it
        print(f"No contours found for {image_path}")
        return None

    # Find the contour with the largest area (presumably the wood log)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a binary mask
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    return mask

# Input and output folders
input_folder = "/content/Chipped_image"
output_folder = "/content/binary_masked"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        # Full path of the input image
        input_path = os.path.join(input_folder, filename)

        # Perform segmentation
        binary_mask = segment_foreground(input_path)

        # Check if segmentation was successful
        if binary_mask is not None:
            # Full path for the output binary mask image
            output_path = os.path.join(output_folder, filename.replace('.', '.'))

            # Save the binary mask image
            cv2.imwrite(output_path, binary_mask)
        else:
            print(f"Skipped processing: {filename}")



# the path to your binary masked images folder
folder_path = "/content/binary_masked"

# Loop through each file in the input folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".png") or file_name.endswith(".jpg"):
        # Read the image
        image_path = os.path.join(folder_path, file_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Darken black regions and brighten white regions
        img = cv2.addWeighted(img, 1.5, img, 0, 0)

        # Overwrite the original image with the processed one
        cv2.imwrite(image_path, img)





'''
#Calculating and storing the result in the shape1.csv file
import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd

# Input and output folders
input_folder = "/content/binary_masked"
output_csv_path = "/content/shape1.csv"

# Create a DataFrame to store the results
df = pd.DataFrame(columns=["Image Name", "Percentage Difference", "Shape"])

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):
        # Full path of the input binary mask image
        input_binary_mask_path = os.path.join(input_folder, filename)

        # Open the binary mask image
        binary_mask = Image.open(input_binary_mask_path)

        # Convert the image to grayscale and then to a NumPy array
        gray_binary_mask = np.array(binary_mask.convert('L'))

        # Find white pixels in the binary mask
        white_pixels = np.column_stack(np.where(gray_binary_mask == 255))

        # Find the first and last white pixel along the diagonals
        first_diagonal_pixel = white_pixels[np.argmin(white_pixels.sum(axis=1))]
        last_diagonal_pixel = white_pixels[np.argmax(white_pixels.sum(axis=1))]

        # Calculate the length of the diagonal line
        diagonal_line_length = int(np.sqrt((last_diagonal_pixel[0] - first_diagonal_pixel[0])**2 + (last_diagonal_pixel[1] - first_diagonal_pixel[1])**2))

        # Find the minimum and maximum coordinates of white pixels
        min_y, min_x = np.min(white_pixels, axis=0)
        max_y, max_x = np.max(white_pixels, axis=0)

        # Calculate the length of the lines in pixels
        vertical_line_length = max_y - min_y
        horizontal_line_length = max_x - min_x

        # Find the smallest line length
        smallest_line_length = min(vertical_line_length, horizontal_line_length, diagonal_line_length)
        #average=(vertical_line_length+ horizontal_line_length+ diagonal_line_length)/3
        # Calculate the radius using the smallest line as diameter
        radius = (smallest_line_length) / 2

        # Calculate the area of a circle using the radius
        area_of_circle = np.pi * (radius**2)

        # Calculate the total number of white pixels
        total_white_pixels = np.sum(gray_binary_mask == 255)

        # Calculate the percentage difference
        per = (abs(total_white_pixels - area_of_circle) / total_white_pixels) * 100

        # Determine the shape based on the percentage difference
        shape = "chapta" if per > 5 else "round"

        # Get the image name without the extension
        image_name = os.path.splitext(filename)[0]

        # Append the result to the DataFrame
        df = df.append({"Image Name": image_name, "Percentage Difference": per, "Shape": shape}, ignore_index=True)

# Save the DataFrame to a CSV file
df.to_csv(output_csv_path, index=False)

'''

#Calculating and storing the result in the shape1.csv file
import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd

# Input and output folders
input_folder = "/content/binary_masked"
output_csv_path = "/content/shape1.csv"

# Create a DataFrame to store the results
df = pd.DataFrame(columns=["Image Name", "Percentage Difference", "Shape"])

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):
        # Full path of the input binary mask image
        input_binary_mask_path = os.path.join(input_folder, filename)

        # Open the binary mask image
        binary_mask = Image.open(input_binary_mask_path)

        # Convert the image to grayscale and then to a NumPy array
        gray_binary_mask = np.array(binary_mask.convert('L'))

        #add

         # Find white pixels in the binary mask
        white_pixels = np.column_stack(np.where(gray_binary_mask == 255))

        # Find the first and last white pixel along the diagonals
        first_diagonal_pixel = white_pixels[np.argmin(white_pixels.sum(axis=1))]
        last_diagonal_pixel = white_pixels[np.argmax(white_pixels.sum(axis=1))]

        # Calculate the length of the diagonal line
        diagonal_line_length = int(np.sqrt((last_diagonal_pixel[0] - first_diagonal_pixel[0])**2 + (last_diagonal_pixel[1] - first_diagonal_pixel[1])**2))

        # Find the minimum and maximum coordinates of white pixels
        min_y, min_x = np.min(white_pixels, axis=0)
        max_y, max_x = np.max(white_pixels, axis=0)

        # Calculate the length of the lines in pixels
        vertical_line_length = max_y - min_y
        horizontal_line_length = max_x - min_x

        # Find the smallest line length
        smallest_line_length = min(vertical_line_length, horizontal_line_length, diagonal_line_length)
        #average=(vertical_line_length+ horizontal_line_length+ diagonal_line_length)/3
        # Calculate the radius using the smallest line as diameter
        radius = (smallest_line_length) / 2

        # Calculate the area of a circle using the radius
        area_of_circle = np.pi * (radius**2)

        # Calculate the total number of white pixels
        total_white_pixels = np.sum(gray_binary_mask == 255)

        #till here

        # Calculate the percentage difference
        per = (abs(total_white_pixels - area_of_circle) / total_white_pixels) * 100

        # Determine the shape based on the percentage difference
        shape = "chapta" if per > 5 else "round"

        # Get the image name without the extension
        image_name = os.path.splitext(filename)[0]

        # **Correct way to add row to DataFrame**
        new_row = pd.DataFrame([[image_name, per, shape]], columns=["Image Name", "Percentage Difference", "Shape"])
        df = pd.concat([df, new_row], ignore_index=True)

# Save the DataFrame to a CSV file
df.to_csv(output_csv_path, index=False)

print("Results saved to", output_csv_path)



#sorting and storing it into the main csv file
def sort_csv(input_file):
    # Read the CSV file into a list of dictionaries
    with open(input_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        data = list(csv_reader)

    # Sort the data based on the values in the "Image Name" column (converted to integers)
    sorted_data = sorted(data, key=lambda x: int(x["Image Name"]))

    # Write the sorted data back to the original CSV file
    with open(input_file, 'w', newline='') as file:
        fieldnames = data[0].keys() if data else []
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(sorted_data)

# '/content/shape1.csv'
sort_csv('/content/shape1.csv')


# Load the original CSV file
original_csv_path = "/content/object_detection_results.csv"
original_df = pd.read_csv(original_csv_path)

# Load the CSV file with the "Defect" column
output_csv_path = "/content/shape1.csv"
output_df = pd.read_csv(output_csv_path)

# Add the "Defect" column to the original DataFrame
original_df['Shape'] = output_df['Shape']

# Save the modified DataFrame back to the original CSV file
original_df.to_csv(original_csv_path, index=False)



'''
-----------------------------------------------------------------
#Log type classification
-----------------------------------------------------------------
'''

# Save the current working directory
current_directory = os.getcwd()

# Change to the desired directory
%cd /content/type_class
print("Current Directory:", current_directory)

import torch
import utils

#deleting if the exp folder already contains
def delete_exp_directory():
    exp_path = '../type_class/runs/predict-cls/exp'

    # Check if the directory exists
    if os.path.exists(exp_path):
        # Remove the directory and its contents
        shutil.rmtree(exp_path)
# Call the function to delete the directory if it exists
delete_exp_directory()

!python classify/predict.py --weights /content/type_class/runs/best.pt --img 128 --source /content/Chipped_image --save-txt

%cd /content
def process_text_file(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, 'r') as file:
        first_row = file.readline().strip().split(' ')

    # Extract the numeric part of the file name without the '.txt' extension
    file_index = os.path.splitext(file_name)[0]

    csv_data = {'Name': file_index, 'Type': first_row[1]}

    return csv_data

folder_path = '/content/type_class/runs/predict-cls/exp/labels'  # Replace with the actual folder path
output_csv_path = '/content/output1.csv'  # Replace with the desired output CSV file path

with open(output_csv_path, 'w', newline='') as csv_file:
    fieldnames = ['Name', 'Type']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write CSV header
    writer.writeheader()

    # Process and sort each text file in the folder
    files_to_process = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.txt')]
    sorted_files = sorted(files_to_process, key=lambda x: int(os.path.splitext(x)[0]))

    for file_name in sorted_files:
        csv_data = process_text_file(folder_path, file_name)
        writer.writerow(csv_data)

print("CSV file created successfully.")

# Load the original CSV file
original_csv_path = "/content/object_detection_results.csv"
original_df = pd.read_csv(original_csv_path)

# Load the CSV file with the "Defect" column
output_csv_path = "/content/output1.csv"
output_df = pd.read_csv(output_csv_path)

# Add the "Defect" column to the original DataFrame
original_df['Type'] = output_df['Type']

# Save the modified DataFrame back to the original CSV file
original_df.to_csv(original_csv_path, index=False)



'''
-----------------------------------------------------------------
#Log defect classification
-----------------------------------------------------------------
'''

# Save the current working directory
current_directory = os.getcwd()

# Change to the desired directory
%cd /content/defects
print("Current Directory:", current_directory)

import torch
import utils

#deleting if the exp folder already contains
def delete_exp_directory():
    exp_path = '../defects/runs/predict-cls/exp'

    # Check if the directory exists
    if os.path.exists(exp_path):
        # Remove the directory and its contents
        shutil.rmtree(exp_path)
# Call the function to delete the directory if it exists
delete_exp_directory()

!python classify/predict.py --weights /content/defects/runs/best.pt --img 128 --source /content/Chipped_image --save-txt

%cd /content
def process_text_file(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, 'r') as file:
        first_row = file.readline().strip().split(' ')

    # Extract the numeric part of the file name without the '.txt' extension
    file_index = os.path.splitext(file_name)[0]

    csv_data = {'Name': file_index, 'Defect': first_row[1]}

    return csv_data

folder_path = '/content/defects/runs/predict-cls/exp/labels'  #  folder path
output_csv_path = '/content/output.csv'  # desired output CSV file path

with open(output_csv_path, 'w', newline='') as csv_file:
    fieldnames = ['Name', 'Defect']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write CSV header
    writer.writeheader()

    # Process and sort each text file in the folder
    files_to_process = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.txt')]
    sorted_files = sorted(files_to_process, key=lambda x: int(os.path.splitext(x)[0]))

    for file_name in sorted_files:
        csv_data = process_text_file(folder_path, file_name)
        writer.writerow(csv_data)

print("CSV file created successfully.")

# Load the original CSV file
original_csv_path = "/content/object_detection_results.csv"
original_df = pd.read_csv(original_csv_path)

# Load the CSV file with the "Defect" column
output_csv_path = "/content/output.csv"
output_df = pd.read_csv(output_csv_path)

# Add the "Defect" column to the original DataFrame
original_df['Defect'] = output_df['Defect']

# Save the modified DataFrame back to the original CSV file
original_df.to_csv(original_csv_path, index=False)



'''
---------------------------------------------------------------------
#image output for each of the attribute logs
---------------------------------------------------------------------
'''

# Load the original CSV file
csv_file_path = "/content/object_detection_results.csv"  # Provide the path to your CSV file
original_df = pd.read_csv(csv_file_path)

# Load the original image
original_image_path = "/content/redline1.jpg"
original_image = cv2.imread(original_image_path)

# Create copies of the original image for each case
copy_circumference = original_image.copy()
copy_defect = original_image.copy()
copy_type = original_image.copy()
copy_shape = original_image.copy()

# Iterate through each row in the CSV file
for index, row in original_df.iterrows():
    left_top = tuple(map(int, eval(row["left top"])))
    bottom_right = tuple(map(int, eval(row["bottom right"])))
    circumference_inches = row["Circumference_inches"]
    defect = row["Defect"]
    log_type = row["Type"]
    shape = row["Shape"]

    # Set bounding box color to red
    box_color = (0, 0, 255)  # Red color

    # Set text color to blue
    text_color = (255, 0, 0)  # Blue color

    # Draw bounding box
    cv2.rectangle(copy_circumference, left_top, bottom_right, box_color, 2)
    cv2.rectangle(copy_defect, left_top, bottom_right, box_color, 2)
    cv2.rectangle(copy_type, left_top, bottom_right, box_color, 2)
    cv2.rectangle(copy_shape, left_top, bottom_right, box_color, 2)

    # Set the font and display text inside the bounding box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    # Calculate text width and height
    text_size = cv2.getTextSize(str(circumference_inches), font, font_scale, font_thickness)[0]
    text_origin = (left_top[0], int((left_top[1] + bottom_right[1] - text_size[1]) / 2))  # Middle left position

    # Draw text
    cv2.putText(copy_circumference, f"{circumference_inches:.2f}", text_origin, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.putText(copy_defect, f"{defect}", text_origin, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.putText(copy_type, f"{log_type}", text_origin, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.putText(copy_shape, f"{shape}", text_origin, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

# Save the copy image with bounding boxes and text
cv2.imwrite("/content/Circumference.jpg", copy_circumference)
cv2.imwrite("/content/Defect.jpg", copy_defect)
cv2.imwrite("/content/Log_type.jpg", copy_type)
cv2.imwrite("/content/Shape.jpg", copy_shape)
