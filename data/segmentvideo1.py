import os
import cv2
import json
import time
import torch
import numpy as np
from PIL import Image
import videoHelper
from sam2.build_sam import build_sam2_video_predictor
import shutil

# Set device and configurations
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = r"checkpoints/sam2_hiera_tiny.pt"
CONFIG = r"sam2_hiera_t.yaml"
predictor = build_sam2_video_predictor(CONFIG, CHECKPOINT)

# Enable torch autocast
torch.autocast(device_type="cuda", dtype=torch.bfloat16)
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Directory setup
OLD_INPUT_DIR = "/home/vijayv/Downloads/Test patches/right/"
NEW_INPUT_DIR = "patch"
OUTPUT_DIR = "segment_images"
MASK_DIR = "segments_masks"
SEGMENT_POINTS_DIR = "segmentation_points"

dirs_to_remove = [NEW_INPUT_DIR, OUTPUT_DIR, MASK_DIR, SEGMENT_POINTS_DIR]

for dir_path in dirs_to_remove:
    try:
        # Remove the directory if it exists
        if os.path.exists(dir_path):
            os.rmdir(dir_path)
            print(f"Removed directory: {dir_path}")
    except OSError as e:
        print(f"Not Found directory {dir_path}")

os.makedirs(NEW_INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(SEGMENT_POINTS_DIR, exist_ok=True)

import os
import cv2
import numpy as np
import json
import time
from PIL import Image



def extract_number(filename):
    try:
        return int(filename.split('_')[1])
    except ValueError:
        return float('inf')

def convert_and_rename_files():
    files = sorted(os.listdir(OLD_INPUT_DIR), key=extract_number)
    
    for idx, filename in enumerate(files):
        original_image_path = os.path.join(OLD_INPUT_DIR, filename)
        image = cv2.imread(original_image_path)
        
        if image is None:
            print(f"Could not read image: {filename}")
            continue
        
        new_filename = f"{idx:05d}.jpg"
        new_image_path = os.path.join(NEW_INPUT_DIR, new_filename)
        
        cv2.imwrite(new_image_path, image)
        print(f"Converted {filename} to {new_filename}")

def load_frame_names(input_dir):
    frame_names = [p for p in os.listdir(input_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    return frame_names

def approximate_contour(contour, target_points=40):
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approximated_contour = cv2.approxPolyDP(contour, epsilon, True)
    
    while len(approximated_contour) > target_points:
        epsilon *= 1.1
        approximated_contour = cv2.approxPolyDP(contour, epsilon, True)
    return approximated_contour

def process_segmentation_results(out_frame_idx, frame_np, video_segments, segment_points_dir, mask_dir, output_dir):
    count = 0
    if out_frame_idx in video_segments:
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            binary_mask = convert_mask_to_binary(out_mask)
            if binary_mask is None:
                continue
            
            outer_points = extract_and_reduce_contour_points(binary_mask)
            save_mask_image(out_frame_idx, out_obj_id+count, binary_mask, frame_np, mask_dir, output_dir)
            save_segmentation_points(out_frame_idx, out_obj_id+count, outer_points, segment_points_dir)
            count += 1

def extract_and_reduce_contour_points(binary_mask, target_points=40):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        approximated_contour = approximate_contour(largest_contour, target_points)
        return approximated_contour.squeeze().tolist()
    return []

def convert_mask_to_binary(out_mask):
    try:
        if out_mask.ndim == 3 and out_mask.shape[0] == 1:
            binary_mask = out_mask[0].astype(np.uint8) * 255
        else:
            print("Warning: Unexpected mask shape.")
            return None

        if binary_mask.ndim != 2 or np.any(np.isnan(binary_mask)) or np.any(np.isinf(binary_mask)):
            print("Warning: Invalid mask dimensions or values.")
            return None

        return binary_mask
    except Exception as e:
        print(f"Error converting mask to binary: {e}")
        return None

def save_segmentation_points(frame_idx, obj_id, outer_points, segment_points_dir):
    image_path = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:05d}obj{obj_id}.png")
    print(f"Attempting to read image from path: {image_path}")
    
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Image at path {image_path} could not be read.")
        image_height = image_width = None
    else:
        image_height, image_width = image.shape[:2]
    
    points_data = {
        "frame_idx": frame_idx,
        "object_id": obj_id,
        "contours": outer_points,
        "imagePath": image_path,
        "image_height": image_height,
        "image_width": image_width
    }
    
    json_output_path = os.path.join(segment_points_dir, f"points_{frame_idx:05d}obj{obj_id}.json")
    
    with open(json_output_path, 'w') as json_file:
        json.dump(points_data, json_file, indent=4)
    
    print(f"Saved segmentation points: {json_output_path}")

def save_mask_image(frame_idx, obj_id, binary_mask, frame_np, mask_dir, output_dir):
    mask_img = Image.fromarray(binary_mask).resize(frame_np.shape[1::-1], Image.NEAREST)
    mask_output_path = os.path.join(mask_dir, f"mask_{frame_idx:05d}obj{obj_id}.png")
    mask_img.save(mask_output_path)

    combined_img = np.where(np.array(mask_img)[..., None] > 0, [255, 0, 0], frame_np).astype(np.uint8)
    output_path = os.path.join(output_dir, f"frame_{frame_idx:05d}obj{obj_id}.png")
    Image.fromarray(combined_img).save(output_path)
    print(f"Saved combined image: {output_path}")

def process_video_frames(frame_names, chunk_size):
    total_frames = len(frame_names)
    for start_frame_idx in range(0, total_frames, chunk_size):
        end_frame_idx = min(start_frame_idx + chunk_size, total_frames)
        start_time = time.time()

        inference_state = predictor.init_state(video_path=NEW_INPUT_DIR)
        predictor.reset_state(inference_state)
        
        annot_frame_path = os.path.join(NEW_INPUT_DIR, frame_names[start_frame_idx])
        points = videoHelper.get_marked_points(annot_frame_path)
        labels = np.array([1  ], np.int32)

        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=start_frame_idx,
            obj_id=1,
            points=points,
            labels=labels,
        )

        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            if start_frame_idx <= out_frame_idx < end_frame_idx:
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

        for out_frame_idx in range(start_frame_idx, end_frame_idx):
            frame_path = os.path.join(NEW_INPUT_DIR, frame_names[out_frame_idx])
            frame_np = np.array(Image.open(frame_path).convert("RGB"))
            process_segmentation_results(out_frame_idx, frame_np, video_segments, SEGMENT_POINTS_DIR, MASK_DIR, OUTPUT_DIR)

        elapsed_time = time.time() - start_time
        print(f"Processing chunk {start_frame_idx // chunk_size + 1} took {elapsed_time:.2f} seconds.")

import math
import json
import glob

def calculate_max_manhattan_distance(filepath):
    # Load the JSON file
    with open(filepath) as f:
        data = json.load(f)

    # Extract the contour points
    contours = data['contours']

    # Initialize variable to track the maximum Manhattan distance
    max_manhattan_distance = 0

    # Iterate through all pairs of points
    for i in range(len(contours)):
        for j in range(i + 1, len(contours)):
            point1 = contours[i]
            point2 = contours[j]

            # Calculate Manhattan distance
            manhattan_distance = abs(point2[0] - point1[0]) + abs(point2[1] - point1[1])

            # Update maximum Manhattan distance if larger
            if manhattan_distance > max_manhattan_distance:
                max_manhattan_distance = manhattan_distance

    return max_manhattan_distance

def calculate_percentage_difference(value1, value2):
    # Calculate the percentage difference
    if value1 == 0:
        return float('inf')  # Avoid division by zero
    return ((value2 - value1) / abs(value1)) * 100

# Get a list of JSON files in the directory



if __name__ == "__main__":
    convert_and_rename_files()
    frame_names = load_frame_names(NEW_INPUT_DIR)    
    process_video_frames(frame_names, chunk_size=len(frame_names))
    json_files = sorted(glob.glob('segmentation_points/*.json'))

        # Initialize variables to store previous maximum Manhattan distance
    previous_max_distance = None

    # Iterate through the JSON files to calculate percentage differences
    for i in range(len(json_files) - 1):
        # Get the current and next file paths
        current_file = json_files[i]
        next_file = json_files[i + 1]

        # Calculate maximum Manhattan distances for both files
        current_max_distance = calculate_max_manhattan_distance(current_file)
        next_max_distance = calculate_max_manhattan_distance(next_file)

        # Calculate percentage difference between the two
        percentage_difference = calculate_percentage_difference(current_max_distance, next_max_distance)

        # Print the results
        print(f"Comparing {current_file} and {next_file}:")
        print(f"  Max Manhattan Distance in {current_file}: {current_max_distance}")
        print(f"  Max Manhattan Distance in {next_file}: {next_max_distance}")
        print(f"  Percentage Difference: {percentage_difference:.2f}%\n")