from PIL import Image

# Импортируем необходимые библиотеки
from ultralyticsplus import YOLO, render_result  
import streamlit as st  # Импортируем streamlit для создания веб-приложения

path_to_model = "best.pt"  # Путь к весам предобученной модели
model = YOLO(path_to_model)  # Инициализируем модель

"""
Функция для обработки изображения и возвращения отрендеренного результата.

Аргументы:
    image_upload (PIL.Image.Image): Загруженное изображение.

Возвращает:
    PIL.Image.Image: Отрендеренное результат изображения.
"""


def process_image(given_image_upload):
    img = Image.open(given_image_upload)  # Открываем загруженное изображение
    results = model.predict(img)  # Предсказываем объекты на изображении
    render = render_result(model=model, image=img, result=results[0])  
    return render  # Возвращаем отрендеренный результат


# Устанавливаем заголовок веб-приложения
st.title("Распознавание объектов на изображении")  

image_upload = st.file_uploader(  # Загружаем изображение
    "Загрузите изображение",
    type=["jpg", "png", "jpeg"]
)

if image_upload is not None:  # Проверяем, было ли загружено изображение
    st.image(  # Отображаем загруженное изображение
        image_upload,
        caption="Загруженное изображение",
        use_column_width=True
    )

    # Отображаем спинер во время обработки изображения
    with st.spinner("Обработка изображения..."):  
        result_image = process_image(image_upload)  # Обрабатываем изображение

    st.image(  # Отображаем обработанное изображение
        result_image,
        caption="Обработанное изображение",
        use_column_width=True
    )
