import unittest
from ultralyticsplus import YOLO


# Определение класса тестирования для обнаружения объектов
class TestObjectDetection(unittest.TestCase):

    # Метод, который выполняется перед каждым тестом
    def setUp(self):
        # Инициализация модели YOLO с предобученными весами "best.pt"
        self.model = YOLO("best.pt")

    # Тест для проверки обнаружения объектов на изображении
    def test_detection(self):
        image_path = "image.jpg"  # Путь к тестовому изображению
        # Получение результатов предсказания модели
        results = self.model.predict(image_path)

        # Проверка, что модель обнаружила хотя бы один объект на изображении
        self.assertGreater(len(results[0].boxes), 0,
                           "Модель не обнаружила объектов на изображении.")

    # Тест для проверки обнаружения объектов любого класса на изображении
    def test_detection_any_class(self):
        image_path = "image.jpg"  # Путь к тестовому изображению
        # Получение результатов предсказания модели
        results = self.model.predict(image_path)

        # Проверка, что модель обнаружила хотя бы один объект на изображении
        self.assertGreater(len(results[0].boxes), 0,
                           "Модель не обнаружила объектов на изображении.")

    def test_detection_image2(self):    
        image_path = "image2.jpg"  # Путь к второму тестовому изображению
        # Получение результатов предсказания модели
        results = self.model.predict(image_path)

        # Проверка, что модель обнаружила хотя бы один объект на изображении
        self.assertGreater(len(results[0].boxes), 0,
                           "Модель не обнаружила объектов на 2 изображении.")

    # Метод, который выполняется после каждого теста
    def tearDown(self):
        pass  # Здесь можно освободить ресурсы, если это необходимо


# Запуск тестов, если этот файл запускается как главный модуль
if __name__ == "__main__":
    unittest.main()
