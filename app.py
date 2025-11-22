from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch
from torchvision import transforms

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model (make sure resnet_best.pth is in same folder)
import timm
import torch.nn as nn

model = timm.create_model("resnet18", pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("resnet_best.pth", map_location=device))
model = model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ['fake', 'real']

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, 1)
    return class_names[idx.item()], conf.item()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"})
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    label, confidence = predict_image(filepath)
    return jsonify({"label": label, "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True)
