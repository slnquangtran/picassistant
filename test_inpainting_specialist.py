import os
import json
from PIL import Image, ImageDraw
from inpainting_specialist import InpaintingSpecialist

def create_test_data(image_path="test_image.png", mask_path="test_mask.png"):
    # Create a dummy image
    img = Image.new('RGB', (512, 512), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    d.rectangle([100, 100, 400, 400], fill=(255, 255, 255))
    img.save(image_path)

    # Create a dummy mask (white square in the middle)
    mask = Image.new('L', (512, 512), color=0)
    dm = ImageDraw.Draw(mask)
    dm.rectangle([100, 100, 400, 400], fill=255)
    mask.save(mask_path)

    return image_path, mask_path

def test_inpainting():
    image_path, mask_path = create_test_data()
    prompt = "A high-quality photo of a cute cat sitting on a field of grass"

    print("Testing Inpainting Specialist...")
    try:
        specialist = InpaintingSpecialist()
        output_json = specialist.run(image_path, mask_path, prompt)
        data = json.loads(output_json)
        
        if "inpainted_image" in data:
            print("Success: Inpainted image received (base64).")
            # To verify, you could save it:
            # from utils import load_image
            # result_img = load_image(data["inpainted_image"])
            # result_img.save("inpainted_result.png")
        else:
            print(f"Failure: Unexpected output structure: {data}")
            
    except Exception as e:
        print(f"Error during testing: {e}")
    finally:
        # Cleanup
        if os.path.exists(image_path):
            os.remove(image_path)
        if os.path.exists(mask_path):
            os.remove(mask_path)

if __name__ == "__main__":
    test_inpainting()
