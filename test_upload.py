import base64
import requests

# Read the image file
with open("Warrior_III_copy.jpeg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

# Send to your Flask service
response = requests.post(
    "http://127.0.0.1:3000/process",
    json={"image_base64": encoded_string}
)

print("Status:", response.status_code)

if response.status_code == 200:
    data = response.json()
    print("Angles:", data["angles"])

    # Save overlay image automatically
    overlay_b64 = data["overlay_base64"]
    overlay_bytes = base64.b64decode(overlay_b64 + '=' * (-len(overlay_b64) % 4))

    with open("overlay_result.png", "wb") as out_file:
        out_file.write(overlay_bytes)

    print("✅ Overlay image saved as overlay_result.png")
else:
    print("Error:", response.text)