import unittest
from ultralyticsplus import YOLO, render_result

class TestObjectDetection(unittest.TestCase):

    def setUp(self):
        self.model = YOLO("best.pt")

    def test_detection(self):
        image_path = "image.jpg"
        results = self.model.predict(image_path)

        self.assertGreater(len(results[0].boxes), 0, "Модель не обнаружила объектов на изображении.")

        for box in results[0].boxes:
            self.assertGreater(box.conf, 0.5, "Уверенность обнаруженного объекта ниже 0.5.")

    def test_detection_any_class(self):
        image_path = "image.jpg"
        results = self.model.predict(image_path)

        self.assertGreater(len(results[0].boxes), 0, "Модель не обнаружила объектов на изображении.")

    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()