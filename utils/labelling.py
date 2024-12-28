import os
import cv2
import pandas as pd
from ultralytics import YOLOWorld
import tqdm

def process_images_and_save_results(input_folder, output_csv, output_image_folder, model_path="yolov8s-worldv2.pt"):
    # Create output directories if they don't exist
    os.makedirs(output_image_folder, exist_ok=True)
    
    # Initialize the YOLO World model
    model = YOLOWorld(model_path)
    model.set_classes(["ball", "robot", "toy", "person"])
    
    # Prepare a list to store results
    results_list = []
    
    # Process all images in the input folder
    for filename in tqdm.tqdm(os.listdir(input_folder)):
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Full path to the image
            image_path = os.path.join(input_folder, filename)
            
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not read image {filename}. Skipping.")
                continue
            
            # Get image height and width
            height, width = img.shape[:2]
            
            # Perform prediction
            results = model(image_path, verbose = False)
            
            # Process each detected object
            for box in results[0].boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Get class and confidence
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                # Draw bounding box on the image
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label with class and confidence
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(img, label, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Store results
                results_list.append({
                    'filename': filename,
                    'width': width,
                    'height': height,
                    'class': class_name,
                    'xmin': x1,
                    'ymin': y1,
                    'xmax': x2,
                    'ymax': y2
                })
            
            # Save the annotated image
            output_image_path = os.path.join(output_image_folder, f"annotated_{filename}")
            cv2.imwrite(output_image_path, img)
    
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(output_csv, index=False)
    
    print(f"Processing complete. Results saved to {output_csv}")
    print(f"Annotated images saved in {output_image_folder}")

# Usage
input_folder = ''
output_csv = ''
output_image_folder = ''

process_images_and_save_results(input_folder, output_csv, output_image_folder)