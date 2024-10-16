import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

points = []  # List to store marked points
drawing = False  # Flag to indicate if a point is being marked

def draw_points(event, x, y, flags, param):
    global drawing, points
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def get_points(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        cap.release()
        return

    # Create a window and set the mouse callback function
    cv2.namedWindow('Video Frame')
    cv2.setMouseCallback('Video Frame', draw_points)

    while True:
        # Display the frame
        img_display = frame.copy()

        # Draw marked points
        for (x, y) in points:
            cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)

        cv2.imshow('Video Frame', img_display)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

    # Print or process the marked points
    print("Marked points:", points)
    return np.array(points, dtype = np.int32)

def save_frames_from_webcam(output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open webcam (0 is the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    frame_count = 0  # Counter for frames

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image.")
            break
        

        
        filename = os.path.join(output_folder, f"{str(frame_count).zfill(5)}.jpg")  # Integer as filename
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        
        frame_count += 1
        
        # Display the frame (optional)
        cv2.imshow('Webcam Frame', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def get_marked_points(image_path):
    drawing = False
    points = []

    def draw_points(event, x, y, flags, param):
        nonlocal drawing
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            points.append((x, y))
        
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    image = cv2.imread(image_path)
    original_shape = image.shape[:2]
    resized_image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_points)

    while True:
        img_display = resized_image.copy()
        for point in points:
            cv2.circle(img_display, point, 3, (0, 255, 0), -1)
        
        cv2.imshow('image', img_display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # Convert points from resized to original dimensions
    x_scale = original_shape[1] / 640
    y_scale = original_shape[0] / 480
    points = [(int(x * x_scale), int(y * y_scale)) for (x, y) in points]

    return np.array(points)