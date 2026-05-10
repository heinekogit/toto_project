import unittest
import numpy as np

from train_hda_multinom import CLASS_ORDER, one_hot, softmax


class TestHdaMultinomClassOrder(unittest.TestCase):
    def test_class_order_is_hda(self):
        self.assertEqual(CLASS_ORDER, ["H", "D", "A"])

    def test_one_hot_column_mapping_hda(self):
        # y_idx: H=0, D=1, A=2 の対応を明示的に確認
        y_idx = np.array([0, 1, 2], dtype=int)
        oh = one_hot(y_idx, 3)
        expected = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(oh, expected)

    def test_softmax_column_alignment(self):
        # 列順 H/D/A で logits を与えたときの最大列が意図どおりか
        logits = np.array(
            [
                [3.0, 1.0, 0.0],  # H最大
                [0.0, 2.0, 1.0],  # D最大
                [1.0, 0.0, 4.0],  # A最大
            ]
        )
        probs = softmax(logits)
        pred_idx = np.argmax(probs, axis=1)
        pred_lbl = [CLASS_ORDER[i] for i in pred_idx]
        self.assertEqual(pred_lbl, ["H", "D", "A"])


if __name__ == "__main__":
    unittest.main()
