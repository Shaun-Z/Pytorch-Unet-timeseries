# prompt: make all the figures in data_gray/imgs to be 1/10 of its original size(resolution) same for the gifs in the mask folder

from PIL import Image
import os

# Define the source and destination directories for images
img_source_dir = 'Pytorch-UNet/data_gray/imgs'
img_dest_dir = 'Pytorch-UNet/data_gray/imgs_resized'

# Create the destination directory if it doesn't exist
os.makedirs(img_dest_dir, exist_ok=True)

# Get a list of image files in the source directory
img_files = [f for f in os.listdir(img_source_dir) if os.path.isfile(os.path.join(img_source_dir, f))]

# Resize images
for file in img_files:
  img_path = os.path.join(img_source_dir, file)
  img = Image.open(img_path)
  width, height = img.size
  new_width = int(width / 10)
  new_height = int(height / 10)
  resized_img = img.resize((new_width, new_height))
  resized_img.save(os.path.join(img_dest_dir, file))


# Define the source and destination directories for masks
mask_source_dir = 'Pytorch-UNet/data_gray/masks'
mask_dest_dir = 'Pytorch-UNet/data_gray/masks_resized'

# Create the destination directory if it doesn't exist
os.makedirs(mask_dest_dir, exist_ok=True)

# Get a list of mask files in the source directory
mask_files = [f for f in os.listdir(mask_source_dir) if os.path.isfile(os.path.join(mask_source_dir, f))]

# Resize masks
for file in mask_files:
  mask_path = os.path.join(mask_source_dir, file)
  img = Image.open(mask_path)
  width, height = img.size
  new_width = int(width / 10)
  new_height = int(height / 10)
  resized_img = img.resize((new_width, new_height))
  resized_img.save(os.path.join(mask_dest_dir, file))



##################################generate random images
from PIL import Image, ImageDraw
import numpy as np
import os
import random

# Define the output folder
output_folder = 'Pytorch-UNet/data_gray/imgs'
os.makedirs(output_folder, exist_ok=True)

# Set the number of images to generate
num_images = 20000
image_width = 192
image_height = 128

# Function to generate a random grayscale image
def generate_random_image(width, height):
    # Randomly choose a generation method
    method = random.choice(['solid', 'gradient', 'shapes'])#'noise', 

    if method == 'noise':
        return generate_random_noise_image(width, height)
    elif method == 'solid':
        return generate_solid_color_image(width, height)
    elif method == 'gradient':
        return generate_gradient_image(width, height)
    elif method == 'shapes':
        return generate_random_shapes_image(width, height)

# Function to generate a random noise image
def generate_random_noise_image(width, height):
    random_data = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    return Image.fromarray(random_data, mode='L')

# Function to generate a solid color image
def generate_solid_color_image(width, height):
    gray_value = random.randint(0, 255)
    return Image.new('L', (width, height), color=gray_value)

# Function to generate a gradient image
def generate_gradient_image(width, height):
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    gradient = np.tile(gradient, (height, 1))
    return Image.fromarray(gradient, mode='L')

# Function to generate an image with random shapes
def generate_random_shapes_image(width, height):
    image = Image.new('L', (width, height), color=255)  # Start with a white background
    draw = ImageDraw.Draw(image)

    # Draw a random number of shapes
    for _ in range(random.randint(1, 5)):
        shape_type = random.choice(['rectangle', 'ellipse', 'line'])

        if shape_type == 'rectangle':
            x0, y0 = random.randint(0, width), random.randint(0, height)
            x1, y1 = random.randint(x0, width), random.randint(y0, height)
            color = random.randint(0, 255)
            draw.rectangle([x0, y0, x1, y1], fill=color)

        elif shape_type == 'ellipse':
            x0, y0 = random.randint(0, width), random.randint(0, height)
            x1, y1 = random.randint(x0, width), random.randint(y0, height)
            color = random.randint(0, 255)
            draw.ellipse([x0, y0, x1, y1], fill=color)

        elif shape_type == 'line':
            x0, y0 = random.randint(0, width), random.randint(0, height)
            x1, y1 = random.randint(0, width), random.randint(0, height)
            color = random.randint(0, 255)
            draw.line([x0, y0, x1, y1], fill=color, width=random.randint(1, 5))

    return image

# Generate and save the images
for i in range(num_images):
    # Generate a random image using one of the methods
    img = generate_random_image(image_width, image_height)

    # Define the output file path
    output_path = os.path.join(output_folder, f'random_image_{i+1}.png')

    # Save the image
    img.save(output_path)

    #


from PIL import Image
import os

# Set the output folder for masks
mask_output_folder = 'Pytorch-UNet/data_gray/masks'
os.makedirs(mask_output_folder, exist_ok=True)

# Set the number of masks to create
num_masks = 20000
mask_width = 192
mask_height = 128

# Function to create a black mask
def create_black_mask(width, height):
    # Create a new black image
    black_mask = Image.new('L', (width, height), 0)  # 'L' mode for grayscale, 0 for black
    return black_mask

# Generate and save masks
for i in range(num_masks):
    # Create a black mask
    mask = create_black_mask(mask_width, mask_height)

    # Define the output file path
    output_path = os.path.join(mask_output_folder, f'random_image_{i+1}_mask.gif')

    # Save the mask as a GIF
    mask.save(output_path, format='GIF')

    # Optional: Print progress every 1000 masks
    if (i + 1) % 1000 == 0:
        print(f'Generated {i + 1} masks')

print(f"Finished generating {num_masks} masks!")

    