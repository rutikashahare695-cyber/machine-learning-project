import os
import sys

# Add src to path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src")
    )
)
from model_inference import load_model


def test_model_loading():
  """Test if the trained model loads successfully."""
  model = load_model()
  assert model is not None, "Model failed to load!"


if __name__ == "__main__":
  test_model_loading()
  print("✅ Unit test passed successfully!")