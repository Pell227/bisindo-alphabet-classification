from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="BISINDO Classification API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model Machine Learning Anda (sesuaikan path model .keras Anda)
MODEL_PATH = "models/mobilenetv2_bisindo.keras" # atau resnet50_bisindo.keras
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Gagal memuat model: {e}")

@app.get("/")
def read_root():
    return {"message": "Backend FastAPI untuk Klasifikasi BISINDO aktif!"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Validasi ekstensi file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")
    
    try:
        # Baca bytes gambar dari request frontend
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocessing gambar sesuai kebutuhan model (contoh ukuran 224x224)
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0) # Tambah dimensi batch (1, 224, 224, 3)
        
        # Prediksi menggunakan model TensorFlow/Keras
        predictions = model.predict(img_array)
        predicted_class_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        # (Opsional) Mapping label kelas sesuai dataset BISINDO Anda
        class_labels = ["A", "B", "C", "D"] # Sesuaikan dengan label asli Anda
        predicted_label = class_labels[predicted_class_idx] if predicted_class_idx < len(class_labels) else str(predicted_class_idx)

        return {
            "success": True,
            "prediction": predicted_label,
            "confidence": round(confidence * 100, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses gambar: {str(e)}")