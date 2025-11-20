#!/usr/bin/env python3
"""
Image normalization script that resizes images to 1024 pixels wide
while maintaining aspect ratio. Processes images from imagedata/ and
saves them to workdata/.
"""

import os
from pathlib import Path
from PIL import Image


def shrink_image(input_path, output_path, target_width=1024):
    """
    Resize an image to target_width while maintaining aspect ratio.
    
    Args:
        input_path: Path to input image
        output_path: Path to save resized image
        target_width: Target width in pixels (default: 1024)
    """
    try:
        with Image.open(input_path) as img:
            # Get original dimensions
            original_width, original_height = img.size
            
            # Calculate new height to maintain aspect ratio
            aspect_ratio = original_height / original_width
            target_height = int(target_width * aspect_ratio)
            
            # Resize image using high-quality resampling
            resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # Save the resized image
            resized_img.save(output_path, quality=95, optimize=True)
            
            print(f"✓ {input_path.name} -> {target_width}x{target_height}")
            
    except Exception as e:
        print(f"✗ Error processing {input_path.name}: {e}")


def main():
    """Main function to process all images in imagedata directory."""
    # Define paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    imagedata_dir = project_dir / "imagedata"
    workdata_dir = project_dir / "workdata"
    
    # Create workdata directory if it doesn't exist
    workdata_dir.mkdir(exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    # Get all image files
    image_files = [
        f for f in imagedata_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No image files found in {imagedata_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Input directory: {imagedata_dir}")
    print(f"Output directory: {workdata_dir}")
    print("-" * 60)
    
    # Process each image
    processed = 0
    for image_file in sorted(image_files):
        output_file = workdata_dir / image_file.name
        shrink_image(image_file, output_file)
        processed += 1
    
    print("-" * 60)
    print(f"Processed {processed} images successfully!")


if __name__ == "__main__":
    main()
