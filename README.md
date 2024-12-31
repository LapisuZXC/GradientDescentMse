# GradientDescentMse

A Python implementation of gradient descent for linear regression using Mean Squared Error (MSE) as the loss function.

## Features
- Implements gradient descent to optimize weights (β) for linear regression.
- Adjustable parameters:
  - Learning rate (`learning_rate`)
  - Convergence threshold (`threshold`)
- Tracks Mean Squared Error (MSE) over iterations.
- Easy-to-use interface for training and analyzing the model.

## Installation Steps

Follow the steps below to set up and run the script:

### 1. Clone the Repository
```bash
git clone https://github.com/LapisuZXC/GradientDescentMse.git
cd GradientDescentMse
```

### 2. Create a Virtual Environment

Set up a Python virtual environment to manage dependencies for your project:

#### On Linux/MacOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Model
The script is designed to optimize linear regression coefficients using gradient descent. You can modify the `GradientDescentMse` parameters in your Python script to explore various configurations.

Here’s a quick example:
```python
import pandas as pd
import numpy as np
from GradientDescentMse import GradientDescentMse  # Adjust import if using a package structure

# Sample data
X = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
y = pd.Series([7, 8, 9])

# Initialize and train the model
model = GradientDescentMse(samples=X, targets=y, learning_rate=0.01, threshold=1e-6)
model.add_constant_feature()
model.learn()

# Access the results
print("Learned weights (beta):", model.beta)
print("MSE progression:", model.iteration_loss_dict)
```

### Visualization (Optional)
You can create a visualization script to analyze the progression of MSE and convergence behavior. Customize the `visualisation.py` script in the repository or create your own.

---

## Customization

Modify the following parameters in the script to experiment with gradient descent:
- `learning_rate`: Controls the step size during weight updates.
- `threshold`: Determines when the algorithm should stop.
- Feature matrix and target values: Use your own data to test the model.

---

For questions, issues, or contributions, refer to the repository or contact the maintainer.

