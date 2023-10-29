#!/bin/bash

# Check if ImageMagick is installed
if ! command -v convert &> /dev/null; then
    echo "ImageMagick is not installed. Install it first."
    exit
fi

# Loop through all PNG files in the current directory
for image in *.png; do
    # Get the filename without the extension
    filename="${image%.*}"

    # Convert the PNG to PDF while preserving the base filename
    convert "$image" "${filename}.pdf"
done

echo "Conversion completed."
