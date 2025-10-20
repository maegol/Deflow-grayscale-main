import os
from PIL import Image, UnidentifiedImageError

# --- Configuration ---

# 1. Set the path to your main repository folder
INPUT_DIR = "/mnt/home_liu/degradation/method_3/dataset"

# 2. Set a path for the new folder where converted images will be saved
OUTPUT_DIR = "/mnt/home_liu/degradation/method_4/DeFlow/Data"

# You can add or remove image formats as needed
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

# ---------------------

def convert_images_to_rgb(input_base_path, output_base_path):
    """
    Recursively finds images in input_base_path, converts them to RGB,
    and saves them in a mirrored directory structure in output_base_path.
    """
    print(f"Starting scan in: {input_base_path}")
    processed_count = 0
    error_count = 0

    # os.walk scans the directory tree top-down
    for dirpath, dirnames, filenames in os.walk(input_base_path):
        for filename in filenames:
            # Check if the file is an image
            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue
            
            # Construct the full path for the input image
            input_image_path = os.path.join(dirpath, filename)

            # --- Create the corresponding output directory ---
            
            # Get the relative path of the folder
            # e.g., "folder1/subfolderA"
            relative_dir = os.path.relpath(dirpath, input_base_path)
            
            # Create the full path for the output directory
            # e.g., "path/to/output_folder/folder1/subfolderA"
            output_dir_path = os.path.join(output_base_path, relative_dir)
            
            # Create the directory if it doesn't exist (and its parents)
            os.makedirs(output_dir_path, exist_ok=True)
            
            # Construct the full path for the output image
            output_image_path = os.path.join(output_dir_path, filename)

            # --- Open, Convert, and Save ---
            try:
                # Use 'with' to ensure the file is closed properly
                with Image.open(input_image_path) as img:
                    
                    # Check if image is already in RGB mode
                    if img.mode == 'RGB':
                        print(f"Skipping (already RGB): {input_image_path}")
                        # We can just copy it or re-save it to be consistent
                        # Let's re-save it to ensure format consistency
                        img.save(output_image_path)
                        continue

                    # This is the core conversion step.
                    # If img.mode is 'L' (grayscale), it will create
                    # an 'RGB' image where R=G=B, just as you described.
                    # It also handles other modes like 'RGBA' (removes alpha)
                    # or 'P' (palette).
                    rgb_img = img.convert('RGB')
                    
                    # Save the new RGB image
                    rgb_img.save(output_image_path)
                    
                    print(f"Converted and saved: {output_image_path}")
                    processed_count += 1

            except (IOError, UnidentifiedImageError) as e:
                print(f"Error processing file {input_image_path}: {e}")
                error_count += 1
            except Exception as e:
                print(f"An unexpected error occurred with {input_image_path}: {e}")
                error_count += 1

    print("\n--- Processing Complete ---")
    print(f"Successfully processed and saved: {processed_count} images.")
    print(f"Failed to process: {error_count} images.")


# This makes the script runnable from the command line
if __name__ == "__main__":
    # Basic check to ensure paths are not the same
    if os.path.abspath(INPUT_DIR) == os.path.abspath(OUTPUT_DIR):
        print("Error: Input and Output directories cannot be the same.")
        print("Please change OUTPUT_DIR to a new location.")
    elif not os.path.isdir(INPUT_DIR):
        print(f"Error: Input directory not found: {INPUT_DIR}")
        print("Please check the INPUT_DIR path.")
    else:
        if not os.path.isdir(OUTPUT_DIR):
            print(f"Output directory not found. Creating: {OUTPUT_DIR}")
            # os.makedirs(OUTPUT_DIR, exist_ok=True) # The main function handles this
        
        convert_images_to_rgb(INPUT_DIR, OUTPUT_DIR)