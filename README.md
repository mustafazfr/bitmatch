# BITMATCH — Screw Detection & Classification

İstanbul Sabahattin Zaim Üniversitesi **BIM 439 Data Mining** dersi projesi (Ocak 2026, Mustafa Zafer).

Elektronik cihazlardaki vidaları fotoğraftan tespit edip tipini (**PH, PZ, SL, T**) sınıflandıran mobil uygulama. İki aşamalı mimari: YOLOv8s ile nesne tespiti → kırpılan vida başı YOLO11m-cls ile sınıflandırma. Mobil taraf Flutter, sunucu tarafı FastAPI.

Detaylı rapor: [`BITMATCH/BITMATCH.pdf`](BITMATCH/BITMATCH.pdf)

## Sonuçlar

- **Tespit (YOLOv8s, 51 epoch):** mAP@50 %96.3, mAP@50-95 %73.7, Precision 0.89, Recall 0.91
- **Sınıflandırma (YOLO11m-cls):** Validasyonda %99.16 Top-1; gerçek dünya testinde (50 yeni görüntü) %85
- SL sınıfı %100 precision/recall; en zor ayrım PH vs PZ (PZ precision %67)

Confusion matrix ve eğitim grafikleri: `BITMATCH/classification_accuracy/`, `BITMATCH/detection_accuracy/`

## Depo Yapısı

| Yol | İçerik |
|-----|--------|
| `BITMATCH/BITMATCH.pdf` | Proje raporu |
| `BITMATCH/codes/backend/` | FastAPI sunucusu (`api.py`) + eğitilmiş modeller (`screw_detector.pt`, `screw_classifier.pt`) |
| `BITMATCH/codes/mobile/screw_detection/` | Flutter mobil uygulama |
| `BITMATCH/classification_accuracy/`, `detection_accuracy/` | Model metrikleri |
| `test_real_world.py`, `test_debug.py` | Gerçek dünya test scriptleri |
| `real_world_confusion_matrix.png` | Gerçek dünya test sonucu |

## Çalıştırma

1. Yerel IP adresinizi öğrenin; mobil uygulamada `main.dart` içindeki `baseUrl` değerini kendi IP'nizle değiştirin.
2. Backend dizininde: `python3 -m uvicorn api:app --host 0.0.0.0 --port 8000`
3. Mobil uygulamayı telefonunuzda çalıştırın (Flutter).

Gereksinimler: Python 3.9+, `ultralytics`, `fastapi`, `opencv-python`, `numpy`; mobil için Flutter SDK.

## Depoya Dahil Edilmeyenler (boyut nedeniyle)

- **Veri setleri** (~6 GB): WDSD ([WEEE Disassembly Screw Dataset](https://vcl.iti.gr/dataset/weee-disassembly-screw-dataset/)), kendi topladığımız ve labelImg ile etiketlediğimiz görüntüler (`my_yolo`, `cls_data`, `enhanced_data` vb.)
- **Eğitim çıktıları** (`runs/`, `runs_det/`, ~2 GB) — final ağırlıklar `codes/backend/models/` altında mevcut
- **Ultralytics baz ağırlıkları** (`yolo11*-cls.pt` vb.) — otomatik indirilir
- **Flutter build artifactları**
