import unittest
from ultralyticsplus import YOLO
import os

class TestYOLOModel(unittest.TestCase):

    def setUp(self):
        self.path_to_model = 'best.pt'
        self.image_path = 'image.jpg'

        self.assertTrue(os.path.exists(self.path_to_model), f"Model file {self.path_to_model} does not exist")

        self.assertTrue(os.path.exists(self.image_path), f"Image file {self.image_path} does not exist")

        self.model = YOLO(self.path_to_model)

    def test_model_prediction(self):
        results = self.model.predict(self.image_path)

        self.assertGreater(len(results), 0, "No predictions were made by the model")

        self.assertGreater(len(results[0].boxes), 0, "No bounding boxes found in the prediction")

        self.assertTrue(self.model.conf is not None, "Model confidence threshold is not set")
        self.assertTrue(self.model.iou is not None, "Model IoU threshold is not set")

if __name__ == '__main__':
    unittest.main()
