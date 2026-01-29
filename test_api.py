import requests
import cv2
import numpy as np
import os
import time

def create_dummy_images():
    # Create a 500x500 green product image
    product = np.zeros((500, 500, 3), dtype=np.uint8)
    product[:] = (0, 255, 0)
    cv2.imwrite("test_product.jpg", product)

    # Create a 100x100 red logo with 50% transparency in alpha channel
    # BGRA
    logo = np.zeros((100, 100, 4), dtype=np.uint8)
    logo[:, :, :3] = (0, 0, 255) # Red
    logo[:, :, 3] = 127 # 50% opacity in the image itself
    cv2.imwrite("test_logo.png", logo)

def test_imprint_api():
    url = "http://127.0.0.1:8000/imprint-logo"
    
    files = {
        "product_image": open("test_product.jpg", "rb"),
        "logo_image": open("test_logo.png", "rb")
    }
    data = {
        "scale": "0.5",
        "opacity": "0.8",
        "position": "center"
    }
    
    try:
        print("Sending request to API...")
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            print("Success! Image received.")
            with open("test_output.png", "wb") as f:
                f.write(response.content)
            print("Saved response to test_output.png")
            
            # Basic validation of output
            out_img = cv2.imread("test_output.png")
            if out_img is not None:
                print(f"Output image shape: {out_img.shape}")
            else:
                print("Failed to decode output image.")
        else:
            print(f"Failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Connection failed: {e}")
    finally:
        files["product_image"].close()
        files["logo_image"].close()
        
        # Cleanup
        if os.path.exists("test_product.jpg"): os.remove("test_product.jpg")
        if os.path.exists("test_logo.png"): os.remove("test_logo.png")

if __name__ == "__main__":
    create_dummy_images()
    # Wait a bit for server to be fully ready if run immediately
    time.sleep(2) 
    test_imprint_api()
