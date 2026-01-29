import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
import os
import uuid

app = FastAPI(title="Image Imprinting API")

# Ensure temp directory exists
TEMP_DIR = "temp_images"
os.makedirs(TEMP_DIR, exist_ok=True)

def cleanup_file(path: str):
    """Deletes the temporary file after response is sent."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"Error cleaning up file {path}: {e}")

async def decode_image(file: UploadFile) -> np.ndarray:
    import cv2
    """Reads an UploadFile and decodes it into an OpenCV image."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {file.filename}")
    return image

def resize_logo_proportional(logo: np.ndarray, product_width: int, scale: float) -> np.ndarray:
    import cv2
    """Resizes the logo based on the product width and scale factor."""
    if scale <= 0 or scale > 1:
        raise HTTPException(status_code=400, detail="Scale must be between 0 and 1.")

    logo_h, logo_w = logo.shape[:2]
    target_width = int(product_width * scale)
    
    if target_width == 0:
         target_width = 1 # Prevent zero division or empty image
         
    aspect_ratio = logo_h / logo_w
    target_height = int(target_width * aspect_ratio)
    
    resized = cv2.resize(logo, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized

def apply_overlay(product: np.ndarray, logo: np.ndarray, position: str, opacity: float) -> np.ndarray:
    import cv2
    """Overlays the logo onto the product image with alpha blending."""
    
    prod_h, prod_w = product.shape[:2]
    logo_h, logo_w = logo.shape[:2]

    # Ensure logo fits (redundant check if resized correctly, but safe)
    if logo_w > prod_w or logo_h > prod_h:
         raise HTTPException(status_code=400, detail="Logo is larger than product image after resizing.")

    # Calculate coordinates
    x_offset, y_offset = 0, 0
    padding = 10 # Optional padding from edges

    if position == "center":
        x_offset = (prod_w - logo_w) // 2
        y_offset = (prod_h - logo_h) // 2
    elif position == "top-left":
        x_offset = padding
        y_offset = padding
    elif position == "top-right":
        x_offset = prod_w - logo_w - padding
        y_offset = padding
    elif position == "bottom-left":
        x_offset = padding
        y_offset = prod_h - logo_h - padding
    elif position == "bottom-right":
        x_offset = prod_w - logo_w - padding
        y_offset = prod_h - logo_h - padding
    else:
        # Default to center if unknown, or raise error. 
        # Requirement said position is string: center | top-left... 
        # I'll default to center for robustness, or valid validation in endpoint
        x_offset = (prod_w - logo_w) // 2
        y_offset = (prod_h - logo_h) // 2
    
    # Ensure offsets are within bounds
    x_offset = max(0, min(x_offset, prod_w - logo_w))
    y_offset = max(0, min(y_offset, prod_h - logo_h))

    # Region of Interest
    roi = product[y_offset:y_offset+logo_h, x_offset:x_offset+logo_w]

    # Normalize alpha
    # If logo has 4 channels (BGRA), extract alpha. 
    # If 3 channels (BGR), assume opaque (alpha=1.0).
    if logo.shape[2] == 4:
        logo_rgb = logo[:, :, :3]
        logo_alpha = logo[:, :, 3] / 255.0
    else:
        logo_rgb = logo
        logo_alpha = np.ones((logo_h, logo_w))

    # Construct alpha mask for blending
    # Global opacity applied to the alpha channel
    alpha_factor = logo_alpha * opacity
    
    # Broadcast alpha to 3 channels for multiplication
    alpha_factor_3c = cv2.merge([alpha_factor, alpha_factor, alpha_factor])

    # Ensure product ROI is 3 channels (if original was 4, might need handling, 
    # but imdecode usually returns BGR unless specified UNCHANGED. 
    # If product is BGRA, we should respect its alpha or convert to BGR for output.
    # For now, let's assume we output the same format as input product if possible, 
    # or convert product to BGR to standardise output.)
    
    roi_bg = roi
    
    # If product has alpha, we need to handle it.
    if roi.shape[2] == 4:
        # If product is transparent, things get complicated. 
        # Simple approach: blend on RGB channels, keep product alpha? 
        # Or blend alpha?
        # Let's operate on RGB part for visual imprint.
        roi_rgb = roi[:, :, :3]
        roi_alpha_channel = roi[:, :, 3]
        
        blended_rgb = (roi_rgb * (1 - alpha_factor_3c) + logo_rgb * alpha_factor_3c)
        
        # Combine back
        product[y_offset:y_offset+logo_h, x_offset:x_offset+logo_w, :3] = blended_rgb.astype(np.uint8)
        # We leave product alpha untouched
    else:
        # Standard BGR
        blended = (roi * (1 - alpha_factor_3c) + logo_rgb * alpha_factor_3c)
        product[y_offset:y_offset+logo_h, x_offset:x_offset+logo_w] = blended.astype(np.uint8)

    return product

@app.post("/imprint-logo")
async def imprint_logo(
    product_image: UploadFile = File(...),
    logo_image: UploadFile = File(...),
    scale: float = Form(0.2),
    opacity: float = Form(0.6),
    position: str = Form("center")
):
    import cv2
    # Validation
    valid_positions = {"center", "top-left", "top-right", "bottom-left", "bottom-right"}
    if position not in valid_positions:
        raise HTTPException(status_code=400, detail=f"Invalid position. Must be one of {valid_positions}")
    
    if not (0.0 < scale <= 1.0):
        # Allow scale up to 1.0 (100% width)
        raise HTTPException(status_code=400, detail="Scale must be between 0.0 and 1.0")
    
    if not (0.0 <= opacity <= 1.0):
         raise HTTPException(status_code=400, detail="Opacity must be between 0.0 and 1.0")

    # Read Images
    prod_img = await decode_image(product_image)
    logo_img = await decode_image(logo_image)

    # Process
    # 1. Resize Logo
    prod_h, prod_w = prod_img.shape[:2]
    logo_resized = resize_logo_proportional(logo_img, prod_w, scale)
    
    # 2. Blend
    result_img = apply_overlay(prod_img, logo_resized, position, opacity)
    
    # Save to temp file
    filename = f"imprinted_{uuid.uuid4()}.png"
    filepath = os.path.join(TEMP_DIR, filename)
    
    success = cv2.imwrite(filepath, result_img)
    if not success:
         raise HTTPException(status_code=500, detail="Failed to save processed image.")
    
    return FileResponse(filepath, media_type="image/png", background=BackgroundTask(cleanup_file, filepath))

def decode_base64_image(base64_str: str) -> np.ndarray:
    import cv2
    import base64

    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    image_bytes = base64.b64decode(base64_str)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    return image

def encode_image_to_base64(image: np.ndarray) -> str:
    import cv2
    import base64

    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image")

    # Return RAW base64 (best for Odoo / APIs)
    return base64.b64encode(buffer).decode("utf-8")

@app.post("/imprint-logo-base64")
async def imprint_logo_base64(
    product_image_base64: str = Body(..., embed=True),
    logo_image_base64: str = Body(..., embed=True),
    scale: float = Body(0.2),
    opacity: float = Body(0.6),
    position: str = Body("center")
):
    import cv2

    valid_positions = {"center", "top-left", "top-right", "bottom-left", "bottom-right"}
    if position not in valid_positions:
        raise HTTPException(status_code=400, detail=f"Invalid position. Must be one of {valid_positions}")

    if not (0.0 < scale <= 1.0):
        raise HTTPException(status_code=400, detail="Scale must be between 0.0 and 1.0")

    if not (0.0 <= opacity <= 1.0):
        raise HTTPException(status_code=400, detail="Opacity must be between 0.0 and 1.0")

    # ✅ Decode base64 → OpenCV images
    prod_img = decode_base64_image(product_image_base64)
    logo_img = decode_base64_image(logo_image_base64)

    # Process (UNCHANGED)
    prod_h, prod_w = prod_img.shape[:2]
    logo_resized = resize_logo_proportional(logo_img, prod_w, scale)
    result_img = apply_overlay(prod_img, logo_resized, position, opacity)

    # ✅ Encode result → base64 (NO FILE SYSTEM)
    result_base64 = encode_image_to_base64(result_img)

    return {
        "image_base64": result_base64
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
