from ..core import Data, LinearRegression

import unittest

import pandas as pd
from sklearn import linear_model


class DataTest(unittest.TestCase):
    def setUp(self):
        self.instance = Data('fixture_data.txt')

    def test_init(self):
        expected = pd.read_csv('fixture_data.txt', delimiter=',')
        self.assertEqual(type(self.instance.data), type(expected))

    def test_number_of_examples(self):
        self.assertEqual(self.instance.number_of_examples(), 2)

    def test_target(self):
        expected = [0, 0]
        self.assertEqual(self.instance.target().all(), all(expected))

    def test_features(self):
        expected = pd.read_csv('fixture_data.txt', delimiter=',').iloc[:, 1:]
        self.assertEqual(self.instance.features().all().all(), expected.all().all())


class LinearRegressionTest(unittest.TestCase):
    def setUp(self):
        self.instance = LinearRegression([[0, 1], [1, 0]], [0, 1])

    def test_train(self):
        expected = linear_model.LinearRegression().fit([[0, 1], [1, 0]], [0, 1])
        self.instance.train()
        self.assertEqual(type(self.instance.model), type(expected))

    def test_coefficients_missing_model(self):
        self.assertEqual(self.instance.coefficients(), None)

    def test_coefficients(self):
        expected = linear_model.LinearRegression().fit([[0, 1], [1, 0]], [0, 1]).coef_
        self.instance.train()
        self.assertEqual(self.instance.coefficients().all(), expected.all())
