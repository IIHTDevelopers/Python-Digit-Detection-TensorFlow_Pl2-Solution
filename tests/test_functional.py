import unittest
import numpy as np
import tensorflow as tf
from main import load_and_preprocess, build_model, train_model, evaluate_model, predict_sample
from tests.TestUtils import TestUtils

class TestMnistClassifierYaksha(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_obj = TestUtils()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = load_and_preprocess()
        cls.model = build_model()
        cls.model = train_model(cls.model, cls.X_train, cls.y_train)

    def test_data_shape_and_normalization(self):
        try:
            shape_correct = self.X_train.shape[1] == 784
            value_range_ok = np.all(self.X_train <= 1.0) and np.all(self.X_train >= 0.0)
            result = shape_correct and value_range_ok
            self.test_obj.yakshaAssert("TestDataShapeAndNormalization", result, "functional")
            print("TestDataShapeAndNormalization =", "Passed" if result else "Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestDataShapeAndNormalization", False, "functional")
            print("TestDataShapeAndNormalization = Failed | Exception:", e)

    def test_model_structure(self):
        try:
            layers = self.model.layers
            dense_layers = [layer for layer in layers if isinstance(layer, tf.keras.layers.Dense)]
            result = (
                dense_layers[0].units == 128 and
                dense_layers[-1].units == 10
            )
            self.test_obj.yakshaAssert("TestModelStructure", result, "functional")
            print("TestModelStructure =", "Passed" if result else "Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestModelStructure", False, "functional")
            print("TestModelStructure = Failed | Exception:", e)

    def test_model_accuracy(self):
        try:
            _, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
            result = accuracy >= 0.85  # Target minimum accuracy
            self.test_obj.yakshaAssert("TestModelAccuracy", result, "functional")
            print("TestModelAccuracy =", "Passed" if result else "Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestModelAccuracy", False, "functional")
            print("TestModelAccuracy = Failed | Exception:", e)

    def test_prediction_output(self):
        try:
            sample = tf.keras.datasets.mnist.load_data()[1][0][0:1]
            sample_image = sample.reshape(1, 28 * 28).astype("float32") / 255.0
            prediction = self.model.predict(sample_image)
            predicted_class = np.argmax(prediction)
            result = 0 <= predicted_class <= 9
            self.test_obj.yakshaAssert("TestPredictionOutput", result, "functional")
            print("TestPredictionOutput =", "Passed" if result else "Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestPredictionOutput", False, "functional")
            print("TestPredictionOutput = Failed | Exception:", e)

