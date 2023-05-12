#!/bin/bash


# Check if the folder path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 path/to/folder"
    exit 1
fi

# Set the folder path
FOLDER_PATH="$1"

# Convert all *.rst files to *.md files
find "$FOLDER_PATH" -type f -iname "*.rst" | while read -r rst_file; do
    md_file="${rst_file%.rst}.md"
    pandoc -s "$rst_file" -o "$md_file" -t gfm
    echo "Converted $rst_file to $md_file"
done

echo "Conversion complete."
