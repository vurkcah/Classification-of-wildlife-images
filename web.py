import streamlit as st
import os
import shutil
import tempfile
from pathlib import Path
import atexit
import patoolib
from ultralytics import YOLO
import threading
import time
#---------------------------------------------------------------------------------------------#
st.set_page_config(
    page_title="Нейросетевой классификатор изображений живой природы",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="collapsed", 
)
#---------------------------------------------------------------------------------------------#
# Функция для загрузки папки с датасетом
def upload_folder():
    uploaded_archive = st.file_uploader("Загрузите папку с датасетом в формате ZIP или RAR", type=["zip", "rar"])
    if uploaded_archive:
        folder_path = os.path.join(os.getcwd(), "uploaded_dataset")
        
        # Очистка папки перед загрузкой нового архива
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path, exist_ok=True)
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            archive_path = os.path.join(tmpdirname, uploaded_archive.name)
            with open(archive_path, "wb") as f:
                f.write(uploaded_archive.getbuffer())
            
            if uploaded_archive.name.endswith(".zip"):
                shutil.unpack_archive(archive_path, folder_path)
            elif uploaded_archive.name.endswith(".rar"):
                patoolib.extract_archive(archive_path, outdir=folder_path)
            
        st.success(f"Папка успешно сохранена в: {folder_path}")
        return folder_path.replace('\\', '/')

    return None
#---------------------------------------------------------------------------------------------#
#Функция для загрузки обученной модели    
def uploaded_model():
    # Создание папки 'valid_model', если она не существует
    model_path = os.path.join(os.getcwd(), "uploaded_model")
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
    uploaded_model = st.file_uploader("Загрузите YOLO модель в формате PT", type=["pt"], key="yolo_model_uploader")
    
    
    if uploaded_model:
        # Очистка папки перед загрузкой нового файла
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        os.makedirs(model_path, exist_ok=True)

        # Сохранение файла .pt
        file_path = os.path.join(model_path, 'model.pt')
        with open(file_path, "wb") as f:
            f.write(uploaded_model.getbuffer())
        
        st.success(f"Модель успешно сохранена в: {model_path}")
        return model_path.replace('\\', '/')

    return None
#---------------------------------------------------------------------------------------------#
# Функция для скачивания обученной модели
def download_trained_model(train_name):
    model_path = f'./runs/classify/{train_name}/'
    archive_path = f'./{train_name}_model.zip'
    if os.path.exists(model_path):
        shutil.make_archive(f'./{train_name}_model', 'zip', model_path)
        with open(archive_path, 'rb') as f:
            st.download_button(
                label="Скачать обученную модель",
                data=f,
                file_name=f'./{train_name}_model.zip',
                mime='application/zip'
            )
    else:
        st.error("Ошибка: Путь к модели не найден. Убедитесь, что обучение прошло успешно.")
#---------------------------------------------------------------------------------------------#
# Функция для очистки папки uploaded_dataset при закрытии приложения
def cleanup_folder(folder_path):
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
    except PermissionError:
        print("Ошибка: Нет прав для удаления папки. Закройте все файлы, которые могут использоваться.")
#---------------------------------------------------------------------------------------------#
#Функция для обучения модели
def train_yolo(dataset_folder_path, progress_indicator, chose_model, chose_epochs, train_name, chose_device):
    model = YOLO(f"yolo11{chose_model}-cls.pt")
    try:
        # Логи выводятся в процессе обучения
        progress_indicator.subheader("Обучение модели началось...🧑🏿‍💻")
        with st.spinner("Обучение модели... Пожалуйста, подождите."):
            model.train(data=f'{dataset_folder_path}/dataset', epochs=chose_epochs, name=train_name, device=chose_device)
        progress_indicator.subheader("Обучение завершено.🦾")
        st.toast('Обучение завершено!', icon='🎉🎉🎉')
        download_trained_model(train_name)
    except Exception as e:
        progress_indicator.subheader("Произошла ошибка во время обучения.")
#---------------------------------------------------------------------------------------------#
#Функция для валидации модели
def valid_yolo(model_path, progress_indicator, dataset_folder_path, name_valid, chose_device):
    model = YOLO(model_path, task='classify')
    try:
        # Логи выводятся в процессе обучения
        progress_indicator.subheader("Валидация модели началась...🧑🏿‍💻")
        with st.spinner("Валидация модели... Пожалуйста, подождите."):
            model.val(data=f'{dataset_folder_path}/dataset', name=name_valid, device=chose_device)
        progress_indicator.subheader("Валидация завершена.📈")
        st.toast('Валидация завершена!', icon='🎉🎉🎉')
    except Exception as e:
        progress_indicator.subheader(f"Произошла ошибка во время валидации: {str(e)}")
#---------------------------------------------------------------------------------------------#
#Функция для классификации изображений
def classify_images(model_path, image_paths, col1, col2, class_names):
    # Проверяем, существует ли путь к модели и изображению
    if not Path(model_path).exists():
        st.error(f"Ошибка: путь к модели '{model_path}' не существует.")
        return
    for image_path in image_paths:
        if not Path(image_path).exists():
            st.error(f"Ошибка: путь к изображению '{image_path}' не существует.")
            return
    # Загружаем модель
    model = YOLO(model_path, task='classify')
    try:
        # Классифицируем изображения
        for image_path in image_paths:
            results = model(image_path)
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(image_path)
            with col2:
                st.subheader("Результаты классификации:")
                for result in results:
                    if result.probs is not None:
                        top5_indices = result.probs.top5
                        top5_confidences = result.probs.top5conf
                        for i in range(5):
                            class_name = class_names[top5_indices[i]] if top5_indices[i] < len(class_names) else "Неизвестный класс"
                            st.write(f"Класс: {class_name}, Вероятность: {top5_confidences[i]:.2f}")
        st.toast('Классификация завершена!', icon='🎉🎉🎉')   
    except Exception as e:
        st.error(f"Произошла ошибка во время классификации: {str(e)}")

#---------------------------------------------------------------------------------------------#
#Функция для загрузки изображений
def upload_images():
    # Определяем путь к папке "uploaded_image"
    upload_folder = 'uploaded_image'
    
    # Проверяем, существует ли папка "uploaded_image", если нет - создаём её
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    # Функция для загрузки изображений
    uploaded_files = st.file_uploader("Загрузите одно или несколько изображений", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    image_paths = []
    if uploaded_files:
        # Удаляем старые файлы из папки "uploaded_image"
        for file_name in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Сохраняем новые загруженные изображения в папку "uploaded_image"
        for uploaded_file in uploaded_files:
            image_path = os.path.join(upload_folder, uploaded_file.name)
            with open(image_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            image_paths.append(image_path)
        st.success(f"Загружено {len(uploaded_files)} изображений.")
        
    return image_paths
#---------------------------------------------------------------------------------------------#
st.header('***Выберите ниже необходимую для вас функцию***')
select_action = st.selectbox("Выберите необходимую для вас функцию",("-- Выберите опцию --","Обучение нейросетевого классификатора", "Валидация нейросетевого классификатора", "Классификация изображений"), label_visibility="hidden", index=0)
#---------------------------------------------------------------------------------------------#
if select_action == "Обучение нейросетевого классификатора":
    st.subheader("Настройте параметры для обучения или оставьте их по умолчанию")
    chose_model = st.select_slider(
        "Выберите размер модели",
        options=[
            "n",
            "s",
            "m",
            "l",
            "x",
        ],
        value="n"
    )
    chose_epochs = st.slider("Выберите количество эпох", 0, 1000, 2)
    chose_device_train = st.select_slider("Выберите способ обучения", options=["cuda", "cpu"], value="cpu")
    if chose_device_train == "cuda":
        chose_device_train=0
    #st.write(f"Валидация {chose_val}")
    train_name = st.text_input("Введите название проекта", "class_nature")
    #st.write("The current movie title is", train_name)
    st.subheader("Загрузите архив с вашим датасетом")
    dataset_folder_path_train = upload_folder()
    if dataset_folder_path_train:
        atexit.register(cleanup_folder, dataset_folder_path_train)
    start_train = st.button("Начать обучение", icon="🚀")
    if start_train:
        if os.path.exists(f"./runs/classify/{train_name}") and os.path.isdir(f"./runs/classify/{train_name}"):
            shutil.rmtree(f"./runs/classify/{train_name}")
        progress_indicator = st.empty()
        # Запуск обучения в основном потоке с анимацией загрузки
        train_yolo(dataset_folder_path_train, progress_indicator, chose_model, chose_epochs, train_name, chose_device_train)
    st.session_state['train_model_path'] = f"./runs/classify/{train_name}/weights/best.pt"
#---------------------------------------------------------------------------------------------#        
if select_action == "Валидация нейросетевого классификатора":
    train_model_path = st.session_state.get('train_model_path', 'Неопределено')
    st.subheader("Настройте параметры для валидации или оставьте их по умолчанию")
    chose_device_valid = st.select_slider("Выберите способ валидации", options=["cuda", "cpu"], value="cpu")
    if chose_device_valid == "cuda":
        chose_device_valid=0
    valid_name = st.text_input("Введите название проекта", "valid_nature")
    st.subheader("Загрузите архив с вашим датасетом")
    dataset_folder_path_valid = upload_folder()
    if dataset_folder_path_valid:
        atexit.register(cleanup_folder, dataset_folder_path_valid)
    chose_model_valid = st.radio("Выберите модель для валидации", ["Текущая обученная модель", "Загрузить новую модель"])
    if chose_model_valid == "Текущая обученная модель":
        if train_model_path == "Неопределено":
           warning = st.warning('Обучите модель!', icon="⚠️")
        else:    
            valid_model = train_model_path
    if chose_model_valid == "Загрузить новую модель":
        uploaded_model = uploaded_model()
        valid_model = f"{uploaded_model}/model.pt"
    valid_start = st.button("Начать валидацию", icon="🚀")
    if valid_start:
        if os.path.exists(f"./runs/classify/{valid_name}") and os.path.isdir(f"./runs/classify/{valid_name}"):
            shutil.rmtree(f"./runs/classify/{valid_name}")
        progress_indicator = st.empty()
        valid_yolo(valid_model, progress_indicator, dataset_folder_path_valid, valid_name, chose_device_valid)
        matrix_path = f"./runs/classify/{valid_name}/confusion_matrix.png"
        matrix_normalized_path = f"./runs/classify/{valid_name}/confusion_matrix_normalized.png"
        if Path(matrix_path).exists():
            st.header("***Результаты валидации***")
            placeholder1 = st.empty()
            placeholder2 = st.empty()
            placeholder1, placeholder2 = st.columns(2)
            with placeholder1:
                st.image(matrix_path)
            with placeholder2:
                st.image(matrix_normalized_path)
#---------------------------------------------------------------------------------------------#              
if select_action == "Классификация изображений":
    train_model_path = st.session_state.get('train_model_path', 'Неопределено')
    class_names = [
    "Цикорий Обыкновенный ",
    "Дневной Павлиний Глаз",
    "Европейский Крот",
    "Жук-Олень",
    "Лебедь Шипун",
    "Тысячелистник"
    ]
    chose_model_class = st.radio("Выберите модель для классификации", ["Текущая обученная модель", "Загрузить новую модель"])
    if chose_model_class == "Текущая обученная модель":
        if train_model_path == "Неопределено":
           warning = st.warning('Обучите модель!', icon="⚠️")
        else:    
            class_model = train_model_path
    if chose_model_class == "Загрузить новую модель":
        uploaded_model = uploaded_model()
        class_model = f"{uploaded_model}/model.pt"
    images = upload_images()
    class_button = st.button("Классифицировать", icon="🚀")
    if class_button:
        placeholder1 = st.empty()
        placeholder2 = st.empty()
        classify_images(class_model, images, placeholder1, placeholder2, class_names)

