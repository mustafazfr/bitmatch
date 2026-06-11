from ultralytics import YOLO
import os
import cv2
import numpy as np
# sklearn varsa import et, yoksa manuel hesapla
try:
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# --- AYARLAR ---
MODEL_PATH = "runs/classify/train/weights/best.pt"
TEST_DIR = "real_world_test"
IMG_SIZE = 1024

def get_center_crop(img, pad_percent=0.05):
    h, w = img.shape[:2]
    pad_h = int(h * pad_percent)
    pad_w = int(w * pad_percent)
    crop = img[pad_h:h-pad_h, pad_w:w-pad_w]
    return crop

def run_test():
    if not os.path.exists(TEST_DIR):
        print(f"❌ HATA: '{TEST_DIR}' klasörü bulunamadı!")
        print("Lütfen masaüstünde 'real_world_test' adında bir klasör oluşturup içine PH, PZ, SL, T klasörlerini koyun.")
        return

    # Sınıfları bul (Sadece klasörleri al, dosyaları yoksay)
    all_items = os.listdir(TEST_DIR)
    class_names = [d for d in all_items if os.path.isdir(os.path.join(TEST_DIR, d)) and not d.startswith('.')]
    class_names.sort()

    print(f"📂 Bulunan Sınıf Klasörleri ({len(class_names)} adet): {class_names}")
    
    if len(class_names) == 0:
        print("❌ HATA: Hiçbir sınıf klasörü bulunamadı! 'real_world_test' klasörünün içine 'PH', 'PZ' vb. klasörler açmalısın.")
        return

    model = YOLO(MODEL_PATH)
    y_true = []
    y_pred = []
    
    total_images = 0

    print("-" * 40)
    for class_name in class_names:
        folder_path = os.path.join(TEST_DIR, class_name)
        files = os.listdir(folder_path)
        
        # Sadece resim dosyalarını filtrele
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heic', '.bmp'))]
        
        print(f"📸 '{class_name}' klasöründe {len(image_files)} resim bulundu.")
        
        for file in image_files:
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"   ⚠️ Uyarı: {file} okunamadı (bozuk olabilir).")
                continue
            
            # Zoom yap
            crop = get_center_crop(img, pad_percent=0.05)
            
            # Tahmin
            results = model.predict(crop, imgsz=IMG_SIZE, verbose=False)
            pred_class = results[0].names[results[0].probs.top1]
            
            y_true.append(class_name)
            y_pred.append(pred_class)
            total_images += 1
            
            if class_name != pred_class:
                print(f"   ❌ HATA: {file} -> Tahmin: {pred_class}")

    if total_images == 0:
        print("\n❌ HATA: Hiçbir resim işlenemedi! Klasörlerin içine resim koyduğundan emin ol.")
        return

    # --- RAPORLAMA ---
    print("\n" + "="*40)
    print("GERÇEK HAYAT TEST SONUÇLARI")
    print("="*40)
    
    # Sklearn varsa detaylı rapor, yoksa basit özet
    if SKLEARN_AVAILABLE:
        # target_names parametresini sadece veri varsa veriyoruz
        unique_labels = sorted(list(set(y_true + y_pred)))
        print(classification_report(y_true, y_pred, labels=unique_labels))
    else:
        # Manuel Basit Rapor
        correct = sum([1 for t, p in zip(y_true, y_pred) if t == p])
        acc = (correct / total_images) * 100
        print(f"Toplam Resim: {total_images}")
        print(f"Doğru Tahmin: {correct}")
        print(f"Başarı Oranı: %{acc:.2f}")

if __name__ == "__main__":
    run_test()
