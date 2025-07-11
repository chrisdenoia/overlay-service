import requests
import base64
import json

# URL of the Warrior III image on Supabase
image_url = "https://hphdlkrpxalmdeneeboi.supabase.co/storage/v1/object/public/processed-data/Warrior_III.jpeg"

# Endpoint to test
endpoint = "https://web-production-4c31.up.railway.app/process"

def download_image(url):
    print("📥 Downloading image...")
    response = requests.get(url)
    response.raise_for_status()
    print(f"✅ Downloaded: {len(response.content)} bytes")
    return response.content

def encode_base64(image_bytes):
    print("🔄 Encoding to base64...")
    return base64.b64encode(image_bytes).decode('utf-8')

def test_endpoint(base64_image):
    print(f"🚀 Sending POST request to {endpoint}...")
    headers = {"Content-Type": "application/json"}
    data = {
        "image_base64": base64_image
    }
    response = requests.post(endpoint, headers=headers, json=data)
    print(f"📡 Response status: {response.status_code}")
    print(f"📋 Headers: {response.headers}")
    print("🧾 Full JSON response:")
    print(json.dumps(response.json(), indent=2))
    return response.json()

def main():
    try:
        img_bytes = download_image(image_url)
        img_base64 = encode_base64(img_bytes)
        result = test_endpoint(img_base64)

        print("\n📊 Summary:")
        print(f"  ✅ Processed: {result.get('processed_successfully')}")
        print(f"  🧍 Pose Detected: {result.get('pose_detected')}")
        print(f"  📌 Landmarks Found: {result.get('landmarks_found')}")
        print(f"  📐 Angles: {result.get('angles')}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()