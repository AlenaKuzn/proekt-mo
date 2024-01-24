import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tkinter import Tk, Frame, Label, Button, Canvas, filedialog
from PIL import Image, ImageTk
import numpy as np


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
    'МАСЛОМ',
    'КОМПЬЮТЕРНАЯ ГРАФИКА',
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


def classify_new_image(image_path, canvas, label_style):
    # Загрузка нового изображения
    new_img = image.load_img(image_path, target_size=(224, 224))
    new_img_array = img_to_array(new_img)

    # Извлечение статистических характеристик цвета
    new_img_features = extract_color_features(new_img_array)

    # Масштабирование при необходимости
    new_img_features_scaled = scaler.transform(np.array([new_img_features]))

    # Предсказание класса
    predicted_class = model.predict(new_img_features_scaled)[0]

    class_names = ["КОМПЬЮТЕРНАЯ ГРАФИКА", "МАСЛОМ", "КАРАНДАШОМ", "ПИКСЕЛЬ-АРТ", "ГЕОМЕТРИЧЕСКИЙ СТИЛЬ",
                   "ФРАКТАЛЬНЫЙ СТИЛЬ"]
    predicted_class_name = class_names[predicted_class]

    # Отображение изображения на Canvas
    img = Image.open(image_path)
    img.thumbnail((224, 224))
    img = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor="nw", image=img)
    canvas.image = img

    # Обновление Label с предсказанным стилем
    label_style.config(text=f"Предсказанный стиль: {predicted_class_name}")


def open_file_dialog(canvas, label_style):
    file_path = filedialog.askopenfilename(title="Выберите изображение",
                                           filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    if file_path:
        classify_new_image(file_path, canvas, label_style)
    else:
        print("Файл не выбран.")


# Создание окна с кнопкой
root_window = Tk()
root_window.title("Классификация изображения")

frame = Frame(root_window)
frame.pack(padx=20, pady=20)

label = Label(frame, text="Нажмите кнопку, чтобы выбрать изображение")
label.pack(pady=10)

# Canvas для отображения изображения
canvas = Canvas(frame, width=224, height=224)
canvas.pack()

# Label для вывода предсказанного стиля
label_style = Label(frame, text="")
label_style.pack(pady=10)

button = Button(frame, text="Выбрать изображение", command=lambda: open_file_dialog(canvas, label_style))
button.pack(pady=10)

root_window.mainloop()
