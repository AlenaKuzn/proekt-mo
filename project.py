import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array


# Функция извлечения статистических характеристик цвета из изображения
def extract_color_features(img_array):
    mean_values = np.mean(img_array, axis=(0, 1)) 
    std_dev_values = np.std(img_array, axis=(0, 1)) 
    features = np.concatenate((mean_values, std_dev_values), axis=-1)
    return features


# Функция обработки изображений в папке и создания данных для обучения
def process_images_in_folder(images_folder, label):
    image_arrays = []
    labels = []

    for filename in os.listdir(images_folder):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            img_path = os.path.join(images_folder, filename)
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img)
            image_arrays.append(img_array)
            labels.append(label)

    return np.array(image_arrays), np.array(labels)

# Папки с изображениями
folders = [
    'КОМПЬЮТЕРНАЯ ГРАФИКА',
    'МАСЛОМ',
    'КАРАНДАШОМ',
    'ПИКСЕЛЬ-АРТ',
    'ГЕОМЕТРИЧЕСКИЙ СТИЛЬ',
    'ФРАКТАЛЬНЫЙ СТИЛЬ'
]

# Подготовка данных для обучения
X = []
y = []

for idx, folder in enumerate(folders):
    images, labels = process_images_in_folder(folder, idx)
    X.extend(images)
    y.extend(labels)

X = np.array(X)
y = np.array(y)

# Извлечение статистических характеристик цвета
X_features = np.array([extract_color_features(img) for img in X])

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение модели (SVM) с изменением ядра на rbf и добавлением параметра C
model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0))
model.fit(X_train_scaled, y_train)

# Предсказание на тестовом наборе
y_pred = model.predict(X_test_scaled)


# Оценка точности модели
#accuracy = accuracy_score(y_test, y_pred)
#print(f"Accuracy: {accuracy}")


def classify_new_image(image_path):
    # Загрузка нового изображения
    new_img = image.load_img(image_path, target_size=(224, 224))
    new_img_array = img_to_array(new_img)

    # Извлечение статистических характеристик цвета
    new_img_features = extract_color_features(new_img_array)

    # Масштабирование при необходимости
    new_img_features_scaled = scaler.transform(np.array([new_img_features]))

    # Предсказание класса
    predicted_class = model.predict(new_img_features_scaled)[0]

    class_names = ["КОМПЬЮТЕРНАЯ ГРАФИКА", "МАСЛОМ", "КАРАНДАШОМ", "ПИКСЕЛЬ-АРТ", "ГЕОМЕТРИЧЕСКИЙ СТИЛЬ", "ФРАКТАЛЬНЫЙ СТИЛЬ"]
    print(f"Предсказанный класс: {class_names[predicted_class]}")

"""
test_1-компьютерная графика
test_2-Пиксель арт
test_3-Маслом
test_4-карандашем
test_5-геометрический стиль
test_6-фрактальный стиль
"""

new_image_path = 'test_5.jpg'
classify_new_image(new_image_path)
