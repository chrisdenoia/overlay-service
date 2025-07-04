import requests
import base64
import json

# Step 1: Call your local overlay service
response = requests.post(
    "http://localhost:3000/process",
    json={"image_base64": open("encoded_input.txt").read().strip()}
)

# Step 2: Parse the JSON
data = response.json()

# Step 3: Save the overlay image
image_base64 = data.get("overlay_base64")
if image_base64:
    # Ensure padding for base64
    missing_padding = len(image_base64) % 4
    if missing_padding:
        image_base64 += '=' * (4 - missing_padding)

    with open("overlay_result.png", "wb") as f:
        f.write(base64.b64decode(image_base64))
    print("✅ Overlay image saved as overlay_result.png")

# Step 4: Print angle feedback
angles = data.get("angles", {})
print("🧘 Angle feedback:")
for joint, angle in angles.items():
    print(f" - {joint.capitalize()}: {angle}°")