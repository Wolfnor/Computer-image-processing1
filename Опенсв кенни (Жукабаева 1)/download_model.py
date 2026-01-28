import urllib.request
import os

print("Скачиваю файлы модели MobileNet-SSD...")

# URL файлов модели
prototxt_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt"
caffemodel_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel"

# Имена файлов
prototxt_file = "MobileNetSSD_deploy.prototxt"
caffemodel_file = "MobileNetSSD_deploy.caffemodel"

# Проверяем, есть ли уже файлы
if os.path.exists(prototxt_file) and os.path.exists(caffemodel_file):
    print("Файлы модели уже существуют!")
else:
    try:
        # Скачиваем prototxt файл
        print("Скачиваю prototxt файл...")
        urllib.request.urlretrieve(prototxt_url, prototxt_file)
        print("✓ prototxt файл скачан")
        
        # Скачиваем caffemodel файл (он большой, около 23 МБ)
        print("Скачиваю caffemodel файл (это может занять время, файл ~23 МБ)...")
        urllib.request.urlretrieve(caffemodel_url, caffemodel_file)
        print("✓ caffemodel файл скачан")
        
        print("\nВсе файлы успешно скачаны! Теперь можно запускать object_detection.py")
    except Exception as e:
        print(f"Ошибка при скачивании: {e}")
        print("\nПопробуйте скачать файлы вручную:")
        print(f"1. {prototxt_url}")
        print(f"2. {caffemodel_url}")

