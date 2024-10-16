import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate polygon area using Shoelace formula
def polygon_area(points):
    x = np.array([point[0] for point in points])
    y = np.array([point[1] for point in points])
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area

# Function to process JSON files and calculate area, transition, growth, and percentage growth
def process_contour_areas(input_folder, output_folder, output_excel):
    areas = {}
    
    # List all JSON files in the input folder
    json_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.json')])
    
    # Loop through each JSON file and calculate area
    for json_file in json_files:
        file_path = os.path.join(input_folder, json_file)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Check if the expected keys are present in the JSON
            if 'frame_idx' in data and 'contours' in data:
                frame_idx = data['frame_idx']
                contours = data['contours']
                area = polygon_area(contours)  # Calculate the area using Shoelace formula
                areas[frame_idx] = area
            else:
                print(f"Skipping file {json_file} due to missing 'frame_idx' or 'contours'.")
    
    # Create DataFrame to store areas and calculate transitions, growth, and percentage growth
    df = pd.DataFrame(list(areas.items()), columns=['Frame', 'Area'])
    df['Transition'] = df['Area'].diff().fillna(0)  # Difference between consecutive frames
    df['Growth'] = df['Area'].cumsum()  # Cumulative growth
    df['Percentage Growth'] = df['Area'].pct_change().fillna(0) * 100  # Percentage growth between frames
    
    # Ensure the output folder exists, create it if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Save the DataFrame to an Excel file, creating the file in the output folder
    output_file_path = os.path.join(output_folder, output_excel)
    with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    
    # Visualization of Area, Transition, Growth, and Percentage Growth over frames
    plt.figure(figsize=(12, 6))
    plt.plot(df['Frame'], df['Area'], label='Area', marker='o')
    plt.plot(df['Frame'], df['Transition'], label='Transition', marker='x')
    plt.plot(df['Frame'], df['Growth'], label='Growth', marker='^')
    
    plt.title('Contour Area, Transition, and Growth Over Frames')
    plt.xlabel('Frame Index')
    plt.ylabel('Pixels')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Return the DataFrame
    return df

# Example usage:
input_folder = r"segmentation_points"  # Replace with your folder
output_folder = r"results"  # Replace with your output file path
output_excel = "new_excel_file.xlsx"  # Name of the output Excel file

# Process the files and calculate the areas
df = process_contour_areas(input_folder, output_folder, output_excel)

# Show the DataFrame
print(df)
