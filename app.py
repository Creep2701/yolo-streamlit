import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cv2
import streamlit as st
import requests
from io import BytesIO
import os
import tempfile


def apply_elliptical_mask(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Define the center, axes lengths and angle of the ellipse
    rows, cols = image.shape
    center = (cols // 2, rows // 2)
    factor = 0.8
    axes_lengths = (int(cols // 2 * factor), int(rows // 2 * factor))

    # Create an elliptical mask with a white ellipse on a black background
    mask = np.zeros_like(image)
    cv2.ellipse(mask, center, axes_lengths, 0, 0, 360, 255, -1)

    # Bitwise-AND operation to keep only the elliptical region
    elliptical_image = cv2.bitwise_and(image, mask)

    # Invert the mask to make the outside white
    mask_inv = cv2.bitwise_not(mask)

    # Combine the elliptical image with the inverted mask to add a white background
    final_image = cv2.bitwise_or(elliptical_image, mask_inv)

    return final_image

def plot_one_box(box, img, color, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def perform_segmentation(segmentation_model_path, image_path):
    segmentation_model = YOLO(segmentation_model_path)
    segmentation_results = segmentation_model.predict(image_path, retina_masks=True)

    # Assuming one primary result
    im_array = segmentation_results[0].plot(line_width=1, conf=0.5, boxes=False)  # BGR numpy array of predictions
    segmentation_image = Image.fromarray(im_array[..., ::-1])  # Convert to RGB PIL image
    return segmentation_image

def preprocess_and_predict(image_path, detection_model_path, segmentation_model_path):
    try:
        # Perform segmentation and get the result image
        target_image = perform_segmentation(segmentation_model_path, image_path)

        # Convert PIL Image to NumPy array in RGB format
        target_image_np = np.array(target_image)

        # Ensure the target image is in the correct format for OpenCV
        # Convert RGB (PIL) to BGR (OpenCV)
        target_image_np = cv2.cvtColor(target_image_np, cv2.COLOR_RGB2BGR)

        # Apply the elliptical mask to preprocess the image
        preprocessed_image = apply_elliptical_mask(image_path)
        preprocessed_image_path = 'temp_preprocessed.png'
        cv2.imwrite(preprocessed_image_path, preprocessed_image)

        # Load the YOLO model for object detection
        detection_model = YOLO(detection_model_path)
        detection_results = detection_model.predict(source=preprocessed_image_path, conf=0.55)

        # Debugging: Print the shape and data type of the image
        print("Target image shape:", target_image_np.shape)
        print("Target image data type:", target_image_np.dtype)

        # Draw bounding boxes on the target image
        for r in detection_results:
            for detection in r.boxes.data:
                x1, y1, x2, y2, conf, cls_id = detection
                label = f'{r.names[int(cls_id)]} {conf:.2f}'
                plot_one_box([x1, y1, x2, y2], target_image_np, label=label, color=(255, 0, 0), line_thickness=2)

        # Convert back to RGB format for display
        final_image = cv2.cvtColor(target_image_np, cv2.COLOR_BGR2RGB)
        final_image = Image.fromarray(final_image)
        
        # Display intermediate results for debugging
        st.image(target_image, caption='Segmented Image', use_column_width=True)
        st.image(Image.fromarray(preprocessed_image), caption='Preprocessed Image', use_column_width=True)
        
        # Display the final processed image
        st.image(final_image, caption='Final Processed Image', use_column_width=True)
        
    except Exception as e:
        # Handle any exceptions that may occur
        print(f"An error occurred: {e}")

# Call preprocess_and_predict within your main function or as needed

   # final_image.save('final_result.jpg')

import streamlit as st
import requests
from io import BytesIO

# Your existing functions here
# ...

def load_image_from_url(url):
    try:
        # Send a GET request to the specified URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Open the image from the byte stream of the response content
            image = Image.open(BytesIO(response.content))
            return image  # Return the image object for further processing
        else:
            # Display an error message if the request was not successful
            st.error("Failed to fetch the image. Status code: " + str(response.status_code))
            return None

    except Exception as e:
        # Display an error message if an exception occurs (e.g., network issues, invalid URL)
        st.error("An error occurred: " + str(e))
        return None


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join("tempDir",uploaded_file.name),"wb") as f:
            f.write(uploaded_file.getbuffer())
        return os.path.join("tempDir",uploaded_file.name)
    except Exception as e:
        return None


def assemble_model_file(parts_folder, original_filename):
    temp_dir = tempfile.mkdtemp()
    assembled_file_path = os.path.join(temp_dir, original_filename)

    with open(assembled_file_path, 'wb') as assembled_file:
        part_number = 1
        while True:
            part_filename = f"{os.path.join(parts_folder, original_filename)}.part{part_number}"
            if not os.path.exists(part_filename):
                break
            with open(part_filename, 'rb') as part_file:
                assembled_file.write(part_file.read())
            part_number += 1

    if part_number == 1:
        st.error("No parts found for assembly. Check the parts folder path.")
        return None

    return assembled_file_path


def main():
    st.title("YOLO Image Processing App")

    parts_folder = 'split_files/'
    segmentation_model_path = assemble_model_file(parts_folder, 'best-segmentation-m.pt')
    detection_model_path = assemble_model_file(parts_folder, 'best-detection-xl.pt')

    if not (segmentation_model_path and detection_model_path):
        st.error("Failed to assemble model files. Please check the file paths.")
        return
    
    option = st.selectbox("How would you like to provide the image?", ['Upload', 'URL'])
    image_path = None

    if option == 'Upload':
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            # Open and convert the uploaded image to JPEG format
            pil_image = Image.open(uploaded_file)
            image_path = "temp_image.jpg"
            pil_image.save(image_path, "JPEG")
            st.image(pil_image, caption='Loaded Image', use_column_width=True)

    elif option == 'URL':
        url = st.text_input("Enter the URL of the image")
        if url:
            image = load_image_from_url(url)
            if image:
                image_path = "temp_image.png"
                image.save(image_path)
                st.image(image, caption='Loaded Image', use_column_width=True)
    if image_path is not None and st.button("Run Model"):
        # Image Processing and Visualization
        processed_image = None
        if image_path:
           # segmentation_model_path = 'best-segmentation-m.pt'
            #detection_model_path = 'best-detection-xl.pt'
            processed_image = preprocess_and_predict(image_path, detection_model_path, segmentation_model_path)

            if isinstance(processed_image, np.ndarray):
                processed_image = Image.fromarray(processed_image)

            if processed_image is not None:
                try:
                    processed_image.save("debug_processed_image.png")
                    st.image(processed_image, caption='Processed Image', use_column_width=True)
                except Exception as e:
                    st.error(f"An error occurred when displaying the image: {e}")

if __name__ == "__main__":
    main()

