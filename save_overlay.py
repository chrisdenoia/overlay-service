import base64

# PASTE your overlay_base64 string between the triple quotes
base64_str = """
<PASTE_STRING_HERE>
"""

# Remove any line breaks or extra whitespace
base64_str = base64_str.strip().replace("\n", "")

# Decode and save the image
with open("overlay_result.png", "wb") as f:
    f.write(base64.b64decode(base64_str))

print("✅ Image saved as overlay_result.png")