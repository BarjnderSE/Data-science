import os

# Define the file structure
project_dir = "flask_id_detector"
templates_dir = os.path.join(project_dir, "templates")
uploads_dir = os.path.join(project_dir, "uploads")

# Create the project directory if it doesn't exist
os.makedirs(project_dir, exist_ok=True)

# Create the templates directory for HTML files
os.makedirs(templates_dir, exist_ok=True)

# Create the uploads directory for storing uploaded images
os.makedirs(uploads_dir, exist_ok=True)

# Define the content for index.html
index_html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ID Tampering Detection</title>
</head>
<body>
    <h1>Upload Your ID for Tampering Detection</h1>
    <form action="/upload" method="POST" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">Upload</button>
    </form>
</body>
</html>"""

# Define the content for result.html
result_html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Detection Result</title>
</head>
<body>
    <h1>Detection Result</h1>
    <p>{{ result }}</p>
    <a href="/">Go Back</a>
</body>
</html>"""

# Write index.html to the templates directory
with open(os.path.join(templates_dir, "index.html"), "w") as f:
    f.write(index_html_content)

# Write result.html to the templates directory
with open(os.path.join(templates_dir, "result.html"), "w") as f:
    f.write(result_html_content)

# Create a Python file for the Flask app (app.py)
flask_app_code = """from flask import Flask, render_template, request, redirect, url_for
from skimage.metrics import structural_similarity
import imutils
import cv2
from PIL import Image
import os
import requests

app = Flask(__name__)

# Function to process image comparison
def detect_tampering(original_image_url, uploaded_image_path):
    original = Image.open(requests.get(original_image_url, stream=True).raw)
    uploaded = Image.open(uploaded_image_path)

    original = original.resize((250, 160))
    uploaded = uploaded.resize((250, 160))

    temp_dir = 'id_tampering_image'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    original_path = os.path.join(temp_dir, 'original.png')
    uploaded_path = os.path.join(temp_dir, 'uploaded.png')

    original.save(original_path)
    uploaded.save(uploaded_path)

    original = cv2.imread(original_path)
    uploaded = cv2.imread(uploaded_path)
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    uploaded_gray = cv2.cvtColor(uploaded, cv2.COLOR_BGR2GRAY)

    score, diff = structural_similarity(original_gray, uploaded_gray, full=True)
    
    os.remove(original_path)
    os.remove(uploaded_path)

    return score * 100

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        uploaded_image_path = os.path.join('uploads', file.filename)
        file.save(uploaded_image_path)

        original_image_url = 'https://www.thestatesman.com/wp-content/uploads/2019/07/pan-card.jpg'

        similarity_score = detect_tampering(original_image_url, uploaded_image_path)

        if similarity_score < 90:
            result = f"Fake ID detected! Similarity score: {similarity_score:.2f}%"
        else:
            result = f"Authentic ID detected! Similarity score: {similarity_score:.2f}%"

        os.remove(uploaded_image_path)

        return render_template('result.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
"""

# Write the Flask app code to app.py
with open(os.path.join(project_dir, "app.py"), "w") as f:
    f.write(flask_app_code)

# Output the folder structure for verification
print("File structure created successfully:")
print(f"Project directory: {project_dir}")
print(f"  ├── app.py")
print(f"  └── templates/")
print(f"      ├── index.html")
print(f"      └── result.html")
print(f"  └── uploads/ (for image uploads)")

