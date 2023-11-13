import imageio
import os
import glob

# Directory containing the PNG files
input_directory = './'
output_video = 'output_video.mp4'

# Video properties
fps = 20  # Frames per second

# Get all PNG files in the directory
png_files = glob.glob(os.path.join(input_directory, 'output_*.png'))
png_files.sort()  # Ensure files are in order

# Create a writer object
writer = imageio.get_writer(output_video, fps=fps)

# Add each image to the video
for file in png_files:
    img = imageio.imread(file)
    writer.append_data(img)

# Close the writer to finish writing the video file
writer.close()

print(f'Movie created successfully: {output_video}')

