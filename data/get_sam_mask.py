import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import glob
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def load_model(model_type="vit_b", checkpoint_path="sam_vit_b_01ec64.pth", device="cuda", image_type="ir"):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    

    if image_type == "ir":
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=64,              
            pred_iou_thresh=0.92,            
            stability_score_thresh=0.95,     
            crop_n_layers=2,                 
            crop_n_points_downscale_factor=1,
            min_mask_region_area=200,        
            points_per_batch=64               
        )

    else:
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=48,              
            pred_iou_thresh=0.88,            
            stability_score_thresh=0.90,     
            crop_n_layers=1,                 
            crop_n_points_downscale_factor=2,
            min_mask_region_area=150,        
            points_per_batch=48               
        )
    
    return mask_generator

def process_image(image_path, mask_generator, device="cuda", image_type="ir"):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    if image_type == "ir":
        if len(image.shape) == 2 or image.shape[2] == 1:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            image = clahe.apply(image)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    masks = mask_generator.generate(image)
    
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    if image_type == "ir":
        for mask in masks:
            if mask["predicted_iou"] > 0.9:
                combined_mask = np.logical_or(combined_mask, mask["segmentation"])
    else:
        for mask in masks:
            combined_mask = np.logical_or(combined_mask, mask["segmentation"])
    
    return combined_mask.astype(np.uint8) * 255

def process_folder(input_folder, output_folder, mask_generator, device="cuda", image_type="ir"):
    os.makedirs(output_folder, exist_ok=True)
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
    
    print(f"Found {len(image_paths)} {image_type.upper()} images to process")
    
    for i, img_path in enumerate(image_paths):
        print(f"\nProcessing {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
        try:
            start_time = cv2.getTickCount()
            
            mask = process_image(img_path, mask_generator, device, image_type)
            if mask is None:
                continue
                
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_folder, f"{name}_mask.png")
            cv2.imwrite(output_path, mask)
            
            elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            print(f"Saved mask to {output_path} | Time: {elapsed_time:.2f}s")
            
            if i % 10 == 0:
                display_result(img_path, mask)
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

def display_result(image_path, mask):
    image = cv2.imread(image_path)
    if image is None:
        return
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='jet')
    plt.title('Segmentation Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 配置参数
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_TYPE = "vit_b"
    CHECKPOINT_PATH = "./sam_vit_b_01ec64.pth"
    
    # ===== 处理红外图像 =====
    IR_INPUT_FOLDER = "./dataset/train/ir"
    IR_OUTPUT_FOLDER = "./dataset/train/ir_sam_mask"

    print(f"\n{'='*40}")
    print(f"Processing INFRARED images")
    print(f"{'='*40}")
    
    ir_mask_generator = load_model(
        model_type=MODEL_TYPE,
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
        image_type="ir"
    )
    
    process_folder(
        input_folder=IR_INPUT_FOLDER,
        output_folder=IR_OUTPUT_FOLDER,
        mask_generator=ir_mask_generator,
        device=DEVICE,
        image_type="ir"
    )
    
    # ===== 处理可见光图像 =====
    VISIBLE_INPUT_FOLDER = "./dataset/train/vis"
    VISIBLE_OUTPUT_FOLDER = "./dataset/train/vi_sam_mask"

    print(f"\n{'='*40}")
    print(f"Processing VISIBLE LIGHT images")
    print(f"{'='*40}")
    
    vi_mask_generator = load_model(
        model_type=MODEL_TYPE,
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
        image_type="vi"
    )
    
    process_folder(
        input_folder=VISIBLE_INPUT_FOLDER,
        output_folder=VISIBLE_OUTPUT_FOLDER,
        mask_generator=vi_mask_generator,
        device=DEVICE,
        image_type="vi"
    )
    
    print("\nProcessing completed for both modalities!")