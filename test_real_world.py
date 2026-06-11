from ultralytics import YOLO
import os
import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- AYARLAR ---
MODEL_PATH = "runs/classify/train12/weights/best.pt"  # Son eğittiğin modelin yolu (kontrol et)
TEST_DIR = "real_world_test"  # Yeni oluşturduğun klasör
IMG_SIZE = 1024  # Modelin eğitim boyutu

def get_center_crop(img, pad_percent=0.05):
    """
    API'deki mantığın aynısı:
    Resmin ortasına zoom yapar (Kenarlardaki %5'lik kısmı atar).
    """
    h, w = img.shape[:2]
    
    # Kırpılacak miktar (Negatif padding mantığı)
    pad_h = int(h * pad_percent)
    pad_w = int(w * pad_percent)
    
    # Kırpma işlemi
    crop = img[pad_h:h-pad_h, pad_w:w-pad_w]
    return crop

def run_test():
    # Modeli yükle
    model = YOLO(MODEL_PATH)
    
    y_true = []
    y_pred = []
    class_names = sorted(os.listdir(TEST_DIR)) # [.DS_Store] varsa silmeyi unutma
    class_names = [c for c in class_names if not c.startswith('.')]
    
    print(f"📂 Sınıflar: {class_names}")
    print("-" * 30)

    for class_name in class_names:
        folder_path = os.path.join(TEST_DIR, class_name)
        if not os.path.isdir(folder_path): continue
        
        files = os.listdir(folder_path)
        print(f"📸 {class_name} sınıfı test ediliyor ({len(files)} resim)...")
        
        for file in files:
            if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
                continue
                
            img_path = os.path.join(folder_path, file)
            
            # Resmi Oku
            img = cv2.imread(img_path)
            if img is None: continue
            
            # API'deki Ön İşlemi Uygula (Center Zoom)
            # Not: Eğer resimlerin çok büyükse önce detection yapmak gerekir ama 
            # sen zaten vida kafasını çekiyorsan bu crop yeterli olur.
            crop = get_center_crop(img, pad_percent=0.05)
            
            # Tahmin
            results = model.predict(crop, imgsz=IMG_SIZE, verbose=False)
            pred_class = results[0].names[results[0].probs.top1]
            
            y_true.append(class_name)
            y_pred.append(pred_class)
            
            # Hataları gör (İsteğe bağlı)
            if class_name != pred_class:
                print(f"   ❌ HATA: {file} -> {class_name} iken {pred_class} tahmin edildi.")

    # --- RAPORLAMA ---
    print("\n" + "="*40)
    print("GERÇEK HAYAT TEST SONUÇLARI")
    print("="*40)
    
    # 1. Genel Rapor
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # 2. Confusion Matrix Çiz
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Olan')
    plt.title('Real-World Confusion Matrix')
    plt.savefig('real_world_confusion_matrix.png')
    print("✅ Confusion Matrix 'real_world_confusion_matrix.png' olarak kaydedildi.")

if __name__ == "__main__":
    run_test()
