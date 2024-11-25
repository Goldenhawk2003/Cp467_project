import cv2
import albumentations as A
import os

def augment_image(image_path, output_folder, num_augments=5):
    # Define augmentations
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.5),
    ])
    
    # Load the image
    image = cv2.imread(image_path)
    image_name = os.path.basename(image_path).split('.')[0]

    # Apply augmentations and save
    for i in range(num_augments):
        augmented = transform(image=image)
        augmented_image = augmented["image"]
        cv2.imwrite(os.path.join(output_folder, f"{image_name}_aug_{i}.jpg"), augmented_image)

# Example usage
input_folder = "/Users/ammarogeil/Downloads/School/Project/Cp467_project/Objects_BG_Removed"
output_folder = "/Users/ammarogeil/Downloads/School/Project/Cp467_project/Augmented_Images" # Replace with your output folder
os.makedirs(output_folder, exist_ok=True)

for img_file in os.listdir(input_folder):
    augment_image(os.path.join(input_folder, img_file), output_folder)
    