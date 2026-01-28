import cv2
import numpy as np
import os

# Проверяем наличие файлов модели
prototxt_file = 'MobileNetSSD_deploy.prototxt'
caffemodel_file = 'MobileNetSSD_deploy.caffemodel'

if not os.path.exists(prototxt_file) or not os.path.exists(caffemodel_file):
    print("ОШИБКА: Файлы модели не найдены!")
    print("Запустите сначала: python download_model.py")
    print("Или скачайте файлы вручную из README.md")
    exit(1)

# Загружаем предобученную модель MobileNet-SSD
# Это готовая модель, которая умеет распознавать 20 разных объектов
print("Загружаю модель...")
net = cv2.dnn.readNetFromCaffe(prototxt_file, caffemodel_file)
print("Модель загружена успешно!")

# Список классов объектов, которые модель может распознать
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Цвета для рамок разных объектов (просто для красоты)
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Открываем камеру (0 - это первая камера)
cap = cv2.VideoCapture(0)

print("Камера запущена! Нажмите 'q' чтобы выйти")

while True:
    # Читаем кадр с камеры
    ret, frame = cap.read()
    
    if not ret:
        print("Не удалось получить кадр с камеры")
        break
    
    # Получаем размеры кадра
    (h, w) = frame.shape[:2]
    
    # Подготавливаем изображение для нейросети
    # MobileNet-SSD работает с изображениями 300x300 пикселей
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
                                  0.007843, (300, 300), 127.5)
    
    # Подаем изображение в нейросеть
    net.setInput(blob)
    detections = net.forward()
    
    # Обрабатываем результаты распознавания
    for i in range(detections.shape[2]):
        # Получаем уверенность (confidence) - насколько модель уверена
        confidence = detections[0, 0, i, 2]
        
        # Если уверенность больше 50%, рисуем рамку
        if confidence > 0.5:
            # Определяем какой это объект
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            
            # Получаем координаты рамки
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Рисуем прямоугольник вокруг объекта
            color = COLORS[idx]
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
            # Пишем название объекта и уверенность
            text = "{}: {:.2f}%".format(label, confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Показываем кадр
    cv2.imshow("Распознавание объектов", frame)
    
    # Если нажали 'q', выходим
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем камеру и закрываем окна
cap.release()
cv2.destroyAllWindows()

