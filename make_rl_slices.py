import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import re

def reorganize_brats_data(source_dir, output_dir, slice_indices=None):
    """
    Reorganize BraTS data by extracting specific slices from each case
    
    Args:
        source_dir: Path to Training PNG folder containing BraTS folders
        output_dir: Path where reorganized data will be saved
        slice_indices: List of slice indices to extract (e.g., [73, 76, 80])
    
    Returns:
        tuple: (number_of_original_folders, number_of_case_groups)
    """
    
    # Default slice indices (middle slices)
    if slice_indices is None:
        slice_indices = [73, 77, 80]  # You can modify these
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all folders in source directory
    all_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
    
    # Group folders by case ID
    case_groups = {}
    for folder in all_folders:
        # Extract case ID from folder name (e.g., "BraTS-GLI-01536-000" from "BraTS-GLI-01536-000-t1c")
        if folder.startswith('BraTS-'):
            case_id = '-'.join(folder.split('-')[:-1])  # Remove the last part (scan type)
            scan_type = folder.split('-')[-1]  # Get scan type (t1c, t1n, t2f, t2w, seg)
            
            if case_id not in case_groups:
                case_groups[case_id] = {}
            case_groups[case_id][scan_type] = folder
    
    print(f"Found {len(case_groups)} unique cases to process")
    print(f"This will reduce {len(all_folders)} folders to {len(case_groups)} organized case folders")
    
    # Safety check
    if len(case_groups) == 0:
        print("Error: No valid BraTS cases found in the source directory!")
        return len(all_folders), 0
    
    # Process each case
    processed_cases = 0
    for case_idx, (case_id, scan_folders) in enumerate(case_groups.items()):
        print(f"Processing case {case_idx + 1}/{len(case_groups)}: {case_id}")
        
        # Create output folder for this case
        case_output_dir = os.path.join(output_dir, f"case_{case_idx:04d}")
        Path(case_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Check if we have all scan types for this case
        expected_types = ['t1c', 't1n', 't2f', 't2w', 'seg']
        available_types = list(scan_folders.keys())
        
        if len(available_types) < 5:
            print(f"  Warning: Only {len(available_types)} scan types found for {case_id}: {available_types}")
        
        # For each slice index we want to extract
        for slice_idx in slice_indices:
            slice_output_dir = os.path.join(case_output_dir, f"slice_{slice_idx:03d}")
            Path(slice_output_dir).mkdir(parents=True, exist_ok=True)
            
            # Extract corresponding slice from each available scan type
            for scan_type, folder_name in scan_folders.items():
                # Path to the vol/images folder
                images_path = os.path.join(source_dir, folder_name, "vol", "images")
                
                if not os.path.exists(images_path):
                    print(f"  Warning: Images path not found: {images_path}")
                    continue
                
                # Get all image files and sort them
                image_files = sorted([f for f in os.listdir(images_path) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                
                if slice_idx < len(image_files):
                    source_file = os.path.join(images_path, image_files[slice_idx])
                    
                    # Map scan type names for consistency
                    scan_type_mapped = map_scan_type(scan_type)
                    dest_file = os.path.join(slice_output_dir, f"{scan_type_mapped}_{slice_idx:03d}.png")
                    
                    # Copy the file
                    try:
                        shutil.copy2(source_file, dest_file)
                    except Exception as e:
                        print(f"  Error copying {source_file}: {e}")
                else:
                    print(f"  Warning: Slice {slice_idx} not available in {scan_type} for {case_id} (only {len(image_files)} images)")
        
        processed_cases += 1
    
    print(f"\nProcessing complete! Processed {processed_cases} cases.")
    
    # Return the counts for summary (moved outside all loops)
    return len(all_folders), len(case_groups)

def map_scan_type(scan_type):
    """Map scan type abbreviations to standard names"""
    mapping = {
        't1c': 'T1C',
        't1n': 'T1N', 
        't2f': 'T2F',
        't2w': 'T2W',
        'seg': 'Segmentation'
    }
    return mapping.get(scan_type.lower(), scan_type.upper())

def create_overlay_images(case_dir, output_overlay_dir):
    """
    Create overlay images combining different scan types
    """
    Path(output_overlay_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all slice folders
    slice_folders = [f for f in os.listdir(case_dir) if f.startswith('slice_')]
    
    for slice_folder in slice_folders:
        slice_path = os.path.join(case_dir, slice_folder)
        
        # Load all images for this slice
        images = {}
        for file in os.listdir(slice_path):
            if file.endswith('.png'):
                scan_type = file.split('_')[0]
                img_path = os.path.join(slice_path, file)
                try:
                    images[scan_type] = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    continue
        
        # Create overlays if we have the required images
        if len(images) >= 2:
            # Example: Overlay T1C with segmentation
            if 'T1C' in images and 'Segmentation' in images:
                overlay = create_colored_overlay(images['T1C'], images['Segmentation'])
                overlay_path = os.path.join(output_overlay_dir, f"{slice_folder}_T1C_with_seg.png")
                cv2.imwrite(overlay_path, overlay)
            
            # Create RGB composite (if you have 3+ modalities)
            if len(images) >= 3:
                rgb_composite = create_rgb_composite(images)
                if rgb_composite is not None:
                    rgb_path = os.path.join(output_overlay_dir, f"{slice_folder}_RGB_composite.png")
                    cv2.imwrite(rgb_path, rgb_composite)

def create_colored_overlay(base_image, overlay_image, alpha=0.3):
    """
    Create a colored overlay of segmentation on base image
    """
    if base_image is None or overlay_image is None:
        return None
        
    # Convert base image to 3-channel
    base_colored = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    
    # Create colored mask (red for segmentation)
    colored_overlay = np.zeros_like(base_colored)
    colored_overlay[:, :, 2] = overlay_image  # Red channel
    
    # Blend images
    result = cv2.addWeighted(base_colored, 1-alpha, colored_overlay, alpha, 0)
    
    return result

def create_rgb_composite(images_dict):
    """
    Create RGB composite from different scan types
    """
    scan_types = list(images_dict.keys())
    
    # Use first 3 scan types for RGB channels (excluding segmentation for now)
    available_scans = [st for st in scan_types if st != 'Segmentation']
    
    if len(available_scans) >= 3:
        r_channel = images_dict[available_scans[0]]
        g_channel = images_dict[available_scans[1]]
        b_channel = images_dict[available_scans[2]]
        
        if r_channel is not None and g_channel is not None and b_channel is not None:
            rgb_composite = cv2.merge([b_channel, g_channel, r_channel])  # BGR format for OpenCV
            return rgb_composite
    
    return None

def analyze_data_structure(source_dir):
    """
    Analyze the data structure to understand what we're working with
    """
    print("Analyzing data structure...")
    
    folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
    print(f"Total folders found: {len(folders)}")
    
    # Group folders by case to see the expected reduction
    case_groups = {}
    for folder in folders:
        if folder.startswith('BraTS-'):
            case_id = '-'.join(folder.split('-')[:-1])
            scan_type = folder.split('-')[-1]
            
            if case_id not in case_groups:
                case_groups[case_id] = []
            case_groups[case_id].append(scan_type)
    
    print(f"Unique cases identified: {len(case_groups)}")
    print(f"Folder reduction: {len(folders)} → {len(case_groups)} folders")
    
    # Show scan type distribution
    scan_type_counts = {}
    for case_id, scan_types in case_groups.items():
        for scan_type in scan_types:
            scan_type_counts[scan_type] = scan_type_counts.get(scan_type, 0) + 1
    
    print("\nScan type distribution:")
    for scan_type, count in sorted(scan_type_counts.items()):
        print(f"  {scan_type}: {count} cases")
    
    # Sample a few folders to understand structure
    sample_folders = folders[:3]
    print(f"\nSample folder structure:")
    
    for folder in sample_folders:
        folder_path = os.path.join(source_dir, folder)
        print(f"\nFolder: {folder}")
        
        # Check for vol folder
        vol_path = os.path.join(folder_path, "vol")
        if os.path.exists(vol_path):
            print(f"  ✓ vol folder found")
            
            # Check for images folder
            images_path = os.path.join(vol_path, "images")
            if os.path.exists(images_path):
                image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                print(f"  ✓ images folder found with {len(image_files)} images")
            else:
                print(f"  ✗ images folder NOT found")
        else:
            print(f"  ✗ vol folder NOT found")
            contents = os.listdir(folder_path)
            print(f"  - Contents: {contents}")

# Example usage
if __name__ == "__main__":
    # First, analyze your data structure
    source_directory = "/Users/arunyahooda/Desktop/BT2O23/Training_png"  # Your original data folder
    
    # Uncomment this line to analyze your data structure first
    # analyze_data_structure(source_directory)
    
    # Set your paths
    output_directory = "/Users/arunyahooda/Desktop/BT2O23/actual_train"  # Where you want the new structure
    
    # Slice indices you want to extract (modify as needed)
    slice_numbers = [73, 76, 80]  # Middle slices, adjust based on your needs
    
    print("Starting data reorganization...")
    
    try:
        # Reorganize the data
        result = reorganize_brats_data(source_directory, output_directory, slice_numbers)
        
        if result is not None and len(result) == 2:
            original_count, case_count = result
            
            print("\nData reorganization complete!")
            
            # Show final summary
            print(f"\nSUMMARY:")
            print(f"Original folders: {original_count}")
            print(f"Organized into: {case_count} case folders")
            print(f"Folder reduction: {original_count} → {case_count} ({original_count - case_count} fewer folders)")
            
            # Count how many slices were extracted
            total_slices = case_count * len(slice_numbers)
            print(f"Total slice combinations created: {total_slices}")
            print(f"Each case now has {len(slice_numbers)} slice folders with up to 5 scan types each")
            
            # Optionally create overlays
            overlay_output = "/Users/arunyahooda/Desktop/BT2O23/Overlay_Images"
            
            if os.path.exists(output_directory):
                case_folders = [f for f in os.listdir(output_directory) if f.startswith('case_')]
                
                if len(case_folders) > 0:
                    print(f"\nCreating overlays for {min(5, len(case_folders))} cases...")
                    for case_folder in case_folders[:5]:  # Process first 5 cases as example
                        case_path = os.path.join(output_directory, case_folder)
                        case_overlay_dir = os.path.join(overlay_output, case_folder)
                        create_overlay_images(case_path, case_overlay_dir)
                    print("Overlay creation complete!")
                else:
                    print("No case folders found to create overlays.")
            
        else:
            print("Error: Function returned invalid result. Check your data structure and paths.")
            
    except Exception as e:
        print(f"Error during processing: {e}")
        print("Please check your source directory path and data structure.")
        import traceback
        traceback.print_exc()
    
    print("\nData reorganization process finished!")