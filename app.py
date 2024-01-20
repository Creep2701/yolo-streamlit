import streamlit as st
from io import BytesIO
import requests
from PIL import Image
import numpy as np
import os
import tempfile
import cv2
from ultralytics import YOLO

# Define the Dropbox links for your model files
segmentation_model_url = "https://www.dropbox.com/scl/fi/f3udyx6kh69pa7zfvtd3g/best-segmentation-medium.pt?rlkey=s1401c70wcj37oklp29khgxkl&dl=0"
detection_model_url = "https://www.dropbox.com/scl/fi/9w73ow1w7mf2o8u6umtp4/best-detection-xlarge.pt?rlkey=g1uutkzrqxh2xlac9s25s0l0m&dl=0"

# Function to download model files
def download_model_files():
    try:
        # Create a temporary directory to store the model files
        temp_dir = tempfile.mkdtemp()

        # Download the segmentation model file
        response_segmentation = requests.get(segmentation_model_url)
        response_segmentation.raise_for_status()
        segmentation_model_path = os.path.join(temp_dir, "best-segmentation-medium.pt")
        with open(segmentation_model_path, "wb") as f:
            f.write(response_segmentation.content)

        # Download the detection model file
        response_detection = requests.get(detection_model_url)
        response_detection.raise_for_status()
        detection_model_path = os.path.join(temp_dir, "best-detection-xlarge.pt")
        with open(detection_model_path, "wb") as f:
            f.write(response_detection.content)

        return segmentation_model_path, detection_model_path

    except Exception as e:
        st.error(f"Failed to download model files: {e}")
        return None, None

# Check if model files exist
def model_files_exist():
    return os.path.isfile("best-segmentation-medium.pt") and os.path.isfile("best-detection-xlarge.pt")

# Function to load an image from a URL
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

def main():
    st.title("YOLO Image Processing App")

    # Check if model files exist
    segmentation_model_path, detection_model_path = model_files_exist()

    if segmentation_model_path is not None and detection_model_path is not None:
        st.success("Model files loaded successfully.")
    else:
        if st.button("Download Model Files"):
            segmentation_model_path, detection_model_path = download_model_files()
            if segmentation_model_path is not None and detection_model_path is not None:
                st.success("Model files downloaded successfully.")
            else:
                st.error("Failed to download model files. Please try again.")

    if segmentation_model_path is not None and detection_model_path is not None:
        # Image upload or URL input
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
                # Load the YOLO model for object detection
                st.write("Loading YOLO model...")
                detection_model = YOLO(detection_model_path)
                detection_results = detection_model.predict(source=image_path, conf=0.55)

                # Rest of your code for processing and visualization...
                # ...
                try:
                    # Perform segmentation and get the result image
                    st.write("Performing segmentation...")
                    target_image = perform_segmentation(segmentation_model_path, image_path)

                    # Convert PIL Image to NumPy array in RGB format
                    target_image_np = np.array(target_image)

                    # Ensure the target image is in the correct format for OpenCV
                    # Convert RGB (PIL) to BGR (OpenCV)
                    target_image_np = cv2.cvtColor(target_image_np, cv2.COLOR_RGB2BGR)

                    # Apply the elliptical mask to preprocess the image
                    st.write("Applying elliptical mask...")
                    preprocessed_image = apply_elliptical_mask(image_path)
                    preprocessed_image_path = 'temp_preprocessed.png'
                    cv2.imwrite(preprocessed_image_path, preprocessed_image)

                    # Debugging: Print the shape and data type of the image
                    st.write("Target image shape:", target_image_np.shape)
                    st.write("Target image data type:", target_image_np.dtype)

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
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
