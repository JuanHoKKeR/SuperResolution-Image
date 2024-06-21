import deeplake
import os
import random
import numpy as np
from PIL import Image, ImageFilter

# Cargar el dataset desde Deeplake
ds = deeplake.load('hub://activeloop/ffhq')[12000:12700]

# Define los directorios donde almacenar las imágenes
base_dir = r'C:\Users\lina-\Documents\ProyectoComputacion\Proyecto\dataset2'
train_low_res_dir = os.path.join(base_dir, 'train/low_res')
train_high_res_dir = os.path.join(base_dir, 'train/high_res')
validation_low_res_dir = os.path.join(base_dir, 'validation/low_res')
validation_high_res_dir = os.path.join(base_dir, 'validation/high_res')
test_low_res_dir = os.path.join(base_dir, 'test/low_res')
test_high_res_dir = os.path.join(base_dir, 'test/high_res')

# Crea los directorios si no existen
os.makedirs(train_low_res_dir, exist_ok=True)
os.makedirs(train_high_res_dir, exist_ok=True)
os.makedirs(validation_low_res_dir, exist_ok=True)
os.makedirs(validation_high_res_dir, exist_ok=True)
os.makedirs(test_low_res_dir, exist_ok=True)
os.makedirs(test_high_res_dir, exist_ok=True)

# Proporciones para entrenamiento, validación y prueba
train_ratio = 0.7
validation_ratio = 0.2
test_ratio = 0.1

# Generar listas de índices aleatorias para entrenamiento, validación y prueba
total_images = len(ds['images_1024/image'])
indices = list(range(total_images))
random.shuffle(indices)

train_end = int(total_images * train_ratio)
validation_end = train_end + int(total_images * validation_ratio)

train_indices = indices[:train_end]
validation_indices = indices[train_end:validation_end]
test_indices = indices[validation_end:]

# Funciones para procesamiento de imágenes
def lower_resolution(image, new_size):
    return image.resize(new_size, Image.Resampling.LANCZOS)

def blur_image(image, radius=1):
    return image.filter(ImageFilter.GaussianBlur(radius))

# Función para guardar la imagen
def save_image(image, path):
    image.save(path)

# Procesar y guardar imágenes
def process_and_save_images(indices, low_res_dir, high_res_dir):
    for index in indices:
        original_image = ds['images_1024/image'][index].numpy()
        high_res_image = Image.fromarray(original_image)

        # Generar y guardar la imagen de alta resolución (256x256)
        mid_res_image = lower_resolution(high_res_image, (256, 256))
        save_image(mid_res_image, os.path.join(high_res_dir, f'image_{index}_256x256.png'))

        # Generar y guardar la imagen de baja resolución (64x64)
        low_res_image = lower_resolution(mid_res_image, (64, 64))
        save_image(low_res_image, os.path.join(low_res_dir, f'image_{index}_64x64.png'))

        # Generar y guardar la imagen de baja resolución difuminada (64x64)
        blurred_image = blur_image(low_res_image, 0.72)
        save_image(blurred_image, os.path.join(low_res_dir, f'image_{index}_64x64_blur0.72.png'))

# Procesar imágenes de entrenamiento, validación y prueba
process_and_save_images(train_indices, train_low_res_dir, train_high_res_dir)
process_and_save_images(validation_indices, validation_low_res_dir, validation_high_res_dir)
process_and_save_images(test_indices, test_low_res_dir, test_high_res_dir)
