import requests
import os

BASE_URL = "http://localhost:8000/api"

def get_test_token():
    # Try login
    res = requests.post(f"{BASE_URL}/auth/login", json={"email": "test@gmail.com", "password": "123456789"})
    if res.status_code == 200:
        return res.json()["token"]
    # Otherwise register
    res = requests.post(f"{BASE_URL}/auth/register", json={"name": "Test Farmer", "email": "test@gmail.com", "password": "123456789"})
    if res.status_code == 200:
        return res.json()["token"]
    raise Exception(f"Failed to auth: {res.text}")

def get_or_create_field(token):
    headers = {"Authorization": f"Bearer {token}"}
    res = requests.get(f"{BASE_URL}/fields", headers=headers)
    fields = res.json().get("fields", [])
    for f in fields:
        if f["name"] == "Advanced Test Field":
            return f["id"]
            
    res = requests.post(f"{BASE_URL}/fields", headers=headers, json={
        "name": "Advanced Test Field", "location": "Test Farm", "area": "1 acre", "soil_type": "Loamy"
    })
    return res.json()["id"]

def get_or_create_crop_plant(token, field_id, crop_name):
    headers = {"Authorization": f"Bearer {token}"}
    res = requests.get(f"{BASE_URL}/crops?field_id={field_id}", headers=headers)
    crops = res.json().get("crops", [])
    crop_id = None
    for c in crops:
        if c["name"] == crop_name:
            crop_id = c["id"]
            break
            
    if not crop_id:
        res = requests.post(f"{BASE_URL}/crops", headers=headers, json={
            "field_id": field_id, "name": crop_name, "variety": "Standard", "season": "Summer", "stage": "Vegetative"
        })
        crop_id = res.json()["id"]
        
    res = requests.get(f"{BASE_URL}/plants?crop_id={crop_id}", headers=headers)
    plants = res.json().get("plants", [])
    plant_id = None
    plant_name = f"{crop_name} Plant 1"
    for p in plants:
        if p["name"] == plant_name:
            plant_id = p["id"]
            break
            
    if not plant_id:
        res = requests.post(f"{BASE_URL}/plants", headers=headers, json={
            "field_id": field_id, "crop_id": crop_id, "name": plant_name
        })
        plant_id = res.json()["id"]
        
    return crop_id, plant_id
