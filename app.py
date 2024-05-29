from PIL import Image
from ultralyticsplus import YOLO, render_result  # Импортируем необходимые библиотеки
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


def process_image(image_upload):
    img = Image.open(image_upload)  # Открываем загруженное изображение
    results = model.predict(img)  # Предсказываем объекты на изображении
    render = render_result(model=model, image=img, result=results[0])  # Рендерим результат
    return render  # Возвращаем отрендеренный результат


st.title("Распознавание объектов на изображении")  # Устанавливаем заголовок веб-приложения

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

    with st.spinner("Обработка изображения..."):  # Отображаем спинер во время обработки изображения
        result_image = process_image(image_upload)  # Обрабатываем изображение

    st.image(  # Отображаем обработанное изображение
        result_image,
        caption="Обработанное изображение",
        use_column_width=True
    )
