import requests
import base64
import os
import glob
from test_utils import get_test_token, get_or_create_field, get_or_create_crop_plant, BASE_URL

CROPS = ["Banana", "Cassava", "Corn", "Cucumber", "Rice", "Soybean", "Tomato"]

def decode_and_save_image(b64_str, filename):
    if not b64_str: return
    if b64_str.startswith("data:image"):
        b64_str = b64_str.split(",")[1]
    with open(filename, "wb") as f:
        f.write(base64.b64decode(b64_str))

def find_test_images(crop_name):
    # Map crop name to folder name
    base_dir = "data/processed/efficientnet/test"
    crop_lower = crop_name.lower()
    
    healthy_folder = f"{base_dir}/{crop_lower}_healthy"
    disease_folders = [f for f in glob.glob(f"{base_dir}/{crop_lower}_*") if not f.endswith("_healthy")]
    
    healthy_img = glob.glob(f"{healthy_folder}/*.jpg") + glob.glob(f"{healthy_folder}/*.jpeg")
    disease_img = []
    if disease_folders:
        disease_img = glob.glob(f"{disease_folders[0]}/*.jpg") + glob.glob(f"{disease_folders[0]}/*.jpeg")
        
    return (healthy_img[0] if healthy_img else None, disease_img[0] if disease_img else None)

def test_disease_advanced():
    print("--- Starting Advanced Disease Detection Tests ---")
    os.makedirs("tests/outputs", exist_ok=True)
    report_path = "tests/disease_report.md"
    
    with open(report_path, "w") as md:
        md.write("# Disease Detection Qualitative Report\n\n")
        
        token = get_test_token()
        headers = {"Authorization": f"Bearer {token}"}
        field_id = get_or_create_field(token)
        
        for i, crop in enumerate(CROPS):
            print(f"Testing {crop}...")
            if i > 0:
                md.write("<div style='page-break-before: always;'></div>\n\n")
            md.write(f"## {crop}\n\n")
            
            crop_id, plant_id = get_or_create_crop_plant(token, field_id, crop)
            
            healthy_img, disease_img = find_test_images(crop)
            
            md.write("| Condition | True Label | Image | GradCAM | Predicted Class | Confidence | Reliable |\n")
            md.write("|-----------|------------|-------|---------|-----------------|------------|----------|\n")
            
            for condition, img_path in [("Healthy", healthy_img), ("Diseased", disease_img)]:
                true_label = os.path.basename(os.path.dirname(img_path)) if img_path else "N/A"
                if not img_path:
                    md.write(f"| {condition} | {true_label} | N/A | N/A | Missing Image | N/A | N/A |\n")
                    continue
                
                with open(img_path, "rb") as f:
                    res = requests.post(
                        f"{BASE_URL}/diagnose",
                        headers=headers,
                        data={"plant_id": plant_id, "crop_id": crop_id, "field_id": field_id},
                        files={"image": (os.path.basename(img_path), f, "image/jpeg")}
                    )
                
                if res.status_code == 200:
                    data = res.json()
                    label = data.get("class_label")
                    conf = data.get("confidence")
                    reliability = data.get("reliability")
                    
                    orig_b64 = data.get("original_image_base64")
                    cam_b64 = data.get("gradcam_image_base64")
                    
                    orig_filename = f"outputs/{crop.lower()}_{condition.lower()}_orig.jpg"
                    cam_filename = f"outputs/{crop.lower()}_{condition.lower()}_cam.jpg"
                    
                    if orig_b64:
                        decode_and_save_image(orig_b64, f"tests/{orig_filename}")
                    else:
                        # Fallback to local test image if API doesn't return base64 (e.g., Unreliable)
                        import shutil
                        shutil.copy(img_path, f"tests/{orig_filename}")
                    if cam_b64:
                        decode_and_save_image(cam_b64, f"tests/{cam_filename}")
                        cam_md = f"<img src='./{orig_filename}' width='150'/>"
                        cam_md2 = f"<img src='./{cam_filename}' width='150'/>"
                    else:
                        cam_md = f"<img src='./{orig_filename}' width='150'/>"
                        cam_md2 = "N/A"
                        
                    md.write(f"| {condition} | **{true_label}** | {cam_md} | {cam_md2} | **{label}** | {conf:.4f} | **{reliability}** |\n")
                    print(f"  [{condition}] -> Predicted: {label} (True: {true_label}) [{conf:.2f}] [{reliability}]")
                else:
                    md.write(f"| {condition} | **{true_label}** | N/A | N/A | API Error | N/A | N/A |\n")
                    print(f"  [{condition}] -> ERROR: {res.text}")
            md.write("\n")
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    test_disease_advanced()
