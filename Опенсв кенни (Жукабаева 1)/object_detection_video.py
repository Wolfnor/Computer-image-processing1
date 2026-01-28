import cv2
import numpy as np
import os
import sys

# Проверяем наличие файлов модели
prototxt_file = 'MobileNetSSD_deploy.prototxt'
caffemodel_file = 'MobileNetSSD_deploy.caffemodel'

if not os.path.exists(prototxt_file) or not os.path.exists(caffemodel_file):
    print("ОШИБКА: Файлы модели не найдены!")
    print("Запустите сначала: python download_model.py")
    exit(1)

# Загружаем предобученную модель MobileNet-SSD
print("Загружаю модель...")
net = cv2.dnn.readNetFromCaffe(prototxt_file, caffemodel_file)
print("Модель загружена успешно!")

# Список классов объектов
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Проверяем аргументы командной строки
if len(sys.argv) > 1:
    video_source = sys.argv[1]  # Путь к видеофайлу
else:
    video_source = 0  # По умолчанию камера

# Открываем видео или камеру
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print(f"Не удалось открыть видео источник: {video_source}")
    exit(1)

print("Видео запущено! Нажмите 'q' чтобы выйти")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Конец видео или ошибка чтения")
        break
    
    (h, w) = frame.shape[:2]
    
    # Подготавливаем изображение для нейросети
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
                                  0.007843, (300, 300), 127.5)
    
    # Подаем изображение в нейросеть
    net.setInput(blob)
    detections = net.forward()
    
    # Обрабатываем результаты
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            color = COLORS[idx]
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
            text = "{}: {:.2f}%".format(label, confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imshow("Распознавание объектов", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

