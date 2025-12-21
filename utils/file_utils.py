import os
import streamlit as st
from utils.image_utils import predict_and_save_image
from utils.video_utils import predict_and_plot_video


def process_media(input_path, output_path , conf_thresh):
    """
    Processes the uploaded media file (image or video) and returns the path to the saved output file.

    Parameters:
    input_path (str): Path to the input media file.
    output_path (str): Path to save the output media file.

    Returns:
    str: The path to the saved output media file.
    """
    file_extension = os.path.splitext(input_path)[1].lower()
    if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        output_path, detection_info = predict_and_plot_video(input_path, output_path)
        return output_path, detection_info

    elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
        output_path, detection_info = predict_and_save_image(input_path, output_path , conf_thresh)
        return output_path, detection_info

    else:
        st.error(f"Unsupported file type: {file_extension}")
        return None, None