import numpy as np
import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

src_path = os.path.join(parent_dir, "src")

sys.path.append(src_path)

from simple_ml.linear_model import LinearRegression


def test_model():
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 2 * X + 5
    y = y.flatten()
    noise = np.random.randn(100) * 0.1
    y = y + noise
    model = LinearRegression(learning_rate=0.01, n_iters=1000)
    model.fit(X, y)
    print(f"real bias {model.bias} and real weights {model.weights}")
    test_x = np.array([[10]])
    predicted_value = model.predict(test_x)
    print(predicted_value)


if __name__ == "__main__":
    test_model()
