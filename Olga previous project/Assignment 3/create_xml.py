#xml file creation

import cv2 as cv
import os

# Ensure the folder exists
os.makedirs('./data', exist_ok=True)

# Create FileStorage object in write mode
fs = cv.FileStorage('./data/checkerboard.xml', cv.FILE_STORAGE_WRITE)

# Write values
fs.write('CheckerBoardWidth', 9)
fs.write('CheckerBoardHeight', 6)
fs.write('CheckerBoardSquareSize', 0.025)

# Release the file
fs.release()

print("checkerboard.xml created successfully.")
