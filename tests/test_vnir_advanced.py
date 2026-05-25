import requests
import base64
import os
import glob
from test_utils import get_test_token, get_or_create_field, get_or_create_crop_plant, BASE_URL

def decode_and_save_image(b64_str, filename):
    if not b64_str: return
    if b64_str.startswith("data:image"):
        b64_str = b64_str.split(",")[1]
    with open(filename, "wb") as f:
        f.write(base64.b64decode(b64_str))

def test_vnir_advanced():
    print("--- Starting Advanced VNIR Stress Monitoring Tests ---")
    os.makedirs("tests/outputs", exist_ok=True)
    report_path = "tests/vnir_report.md"
    
    with open(report_path, "w") as md:
        md.write("# VNIR Stress Monitoring Qualitative Report\n\n")
        
        token = get_test_token()
        headers = {"Authorization": f"Bearer {token}"}
        field_id = get_or_create_field(token)
        crop_id, plant_id = get_or_create_crop_plant(token, field_id, "Banana")
        
        md.write("## Phase 1: Baseline Calibration (Banana)\n\n")
        md.write("Uploading 5 healthy banana images to establish the baseline.\n\n")
        
        md.write("| Scan | Status | Ratio | Baseline | Image | HSV Isolate | NIR Predicted |\n")
        md.write("|------|--------|-------|----------|-------|-------------|---------------|\n")
        
        # Get 5 healthy
        base_dir = "data/processed/efficientnet/test"
        healthy_imgs = (glob.glob(f"{base_dir}/banana_healthy/*.jpg") + glob.glob(f"{base_dir}/banana_healthy/*.jpeg"))[:5]
        
        for i, img_path in enumerate(healthy_imgs):
            with open(img_path, "rb") as f:
                res = requests.post(
                    f"{BASE_URL}/vnir-upload",
                    headers=headers,
                    data={"plant_id": plant_id, "crop_id": crop_id, "field_id": field_id},
                    files={"image": (os.path.basename(img_path), f, "image/jpeg")}
                )
            if res.status_code == 200:
                data = res.json()
                hsv_b64 = data.get("hsv_image_base64")
                nir_b64 = data.get("vnir_image_base64")
                
                base_name = f"vnir_calib_{i}"
                import shutil
                shutil.copy(img_path, f"tests/outputs/{base_name}_orig.jpg")
                if hsv_b64: decode_and_save_image(hsv_b64, f"tests/outputs/{base_name}_hsv.jpg")
                if nir_b64: decode_and_save_image(nir_b64, f"tests/outputs/{base_name}_nir.jpg")
                
                md.write(f"| {i+1} | {data.get('status')} | {data.get('ratio'):.4f} | {data.get('baseline') or 'N/A'} | <img src='./outputs/{base_name}_orig.jpg' width='100'/> | <img src='./outputs/{base_name}_hsv.jpg' width='100'/> | <img src='./outputs/{base_name}_nir.jpg' width='100'/> |\n")
                print(f"  Calib {i+1} -> {data.get('status')} (Ratio: {data.get('ratio'):.4f})")
        
        md.write("\n<div style='page-break-before: always;'></div>\n\n## Phase 2: Stress Detection (Banana Sigatoka)\n\n")
        md.write("Testing with 3 diseased (Sigatoka) images.\n\n")
        
        md.write("| Scan | Status | Ratio | Vs Baseline | Image | HSV Isolate | NIR Predicted |\n")
        md.write("|------|--------|-------|-------------|-------|-------------|---------------|\n")
        
        sigatoka_imgs = (glob.glob(f"{base_dir}/banana_sigatoka/*.jpg") + glob.glob(f"{base_dir}/banana_sigatoka/*.jpeg"))[:3]
        
        for i, img_path in enumerate(sigatoka_imgs):
            with open(img_path, "rb") as f:
                res = requests.post(
                    f"{BASE_URL}/vnir-upload",
                    headers=headers,
                    data={"plant_id": plant_id, "crop_id": crop_id, "field_id": field_id},
                    files={"image": (os.path.basename(img_path), f, "image/jpeg")}
                )
            if res.status_code == 200:
                data = res.json()
                hsv_b64 = data.get("hsv_image_base64")
                nir_b64 = data.get("vnir_image_base64")
                
                base_name = f"vnir_stress_{i}"
                import shutil
                shutil.copy(img_path, f"tests/outputs/{base_name}_orig.jpg")
                if hsv_b64: decode_and_save_image(hsv_b64, f"tests/outputs/{base_name}_hsv.jpg")
                if nir_b64: decode_and_save_image(nir_b64, f"tests/outputs/{base_name}_nir.jpg")
                
                vs_b = data.get('vs_baseline')
                vs_b_str = f"{vs_b:.2f}%" if vs_b is not None else "N/A"
                
                md.write(f"| {i+1} | **{data.get('status')}** | {data.get('ratio'):.4f} | {vs_b_str} | <img src='./outputs/{base_name}_orig.jpg' width='100'/> | <img src='./outputs/{base_name}_hsv.jpg' width='100'/> | <img src='./outputs/{base_name}_nir.jpg' width='100'/> |\n")
                print(f"  Stress {i+1} -> {data.get('status')} (Ratio: {data.get('ratio'):.4f}, vs: {vs_b_str})")
        
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    test_vnir_advanced()
