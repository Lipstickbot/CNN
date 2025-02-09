import os
import dlib
from skimage import io
from scipy.spatial import distance
import cv2


# Работа с dlib
sp = dlib.shape_predictor('c:\\Users\\vovot\\Downloads\\shape_predictor_68_face_landmarks.dat')
recover = dlib.face_recognition_model_v1('c:\\Users\\vovot\\Downloads\\dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

# Съемка с камеры
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Камера не открыта')
    exit()

print("Нажмите 'q', чтобы сделать снимок и начать сравнение.")
frame = None
while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удается получить кадр. Завершение ...")
        break

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        print("Фото сделано!")
        break

cap.release()
cv2.destroyAllWindows()

# Преобразование кадра в формат dlib
img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Обнаружение лица в кадре
dets_web = detector(img2, 1)
if len(dets_web) == 0:
    print("Лицо на фото с камеры не обнаружено.")
    exit()

shape2 = sp(img2, dets_web[0])
face_dest2 = recover.compute_face_descriptor(img2, shape2)

# Путь к папке с фотографиями
photos_dir = 'c:\\Users\\vovot\\OneDrive\\Pictures\\Camera Roll\\'

# Инициализация переменных для сравнения
min_distance = float('inf')
best_match = None

# Перебор всех фотографий из папки
for filename in os.listdir(photos_dir):
    photo_path = os.path.join(photos_dir, filename)

    try:
        img1 = io.imread(photo_path)
        dets = detector(img1, 1)

        if len(dets) == 0:
            print(f"Лицо на фото {filename} не обнаружено. Пропуск.")
            continue

        shape1 = sp(img1, dets[0])
        face_dest1 = recover.compute_face_descriptor(img1, shape1)

        # Вычисление Евклидова расстояния
        dist = distance.euclidean(face_dest1, face_dest2)
        print(f"Фото: {filename}, Евклидово расстояние: {dist}")
        

        if dist < min_distance:
            min_distance = dist
            best_match = filename

    except Exception as e:
        print(f"Ошибка при обработке {filename}: {e}")
        continue

# Результат
if best_match:
    print(f"Лучшее совпадение: {best_match}, минимальное расстояние: {min_distance}")
else:
    print("Совпадений не найдено.")
