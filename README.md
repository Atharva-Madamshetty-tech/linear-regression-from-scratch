# Linear Regression from Scratch (Python, No Sklearn)

This repository contains multiple projects where I **built Linear Regression models from scratch using only Python, NumPy, and basic math — without relying on libraries like Scikit-learn**.  
The goal is to understand the **fundamentals of machine learning** by implementing the core concepts (gradient descent, cost function, weight updates) manually.

---

## 📂 Project Structure

linear-regression-from-scratch/
├── single_variable/
│ ├── house_price.py # Predict house prices using one feature
│ ├── student_marks.py # Predict student marks using study hours
├── multiple_variable/
│ ├── house_price.py # Predict house prices using multiple features
│ ├── wine_quality.py # Predict wine quality using physicochemical data
├── README.md # Project documentation
└── requirements.txt # Dependencies (NumPy, Matplotlib)


---

## 🔍 About the Projects

### 1. 🏠 House Price Predictor
- **Single-variable version**: Predicts house price based on a single feature (e.g., size of the house).  
- **Multi-variable version**: Uses multiple features (e.g., size, location index, number of rooms) for better predictions.  
- Demonstrates how adding features improves accuracy.

### 2. 📚 Student Marks Predictor (Single-variable)
- Predicts student marks based on the number of study hours.  
- A simple introduction to Linear Regression, perfect for visualizing the line of best fit.  
- Great example to explain the role of **cost function (MSE)** and **gradient descent**.

### 3. 🍷 Wine Quality Prediction (Multi-variable)
- Predicts wine quality based on physicochemical properties (acidity, sugar, alcohol, etc.).  
- A more **real-world dataset** where multiple variables influence the output.  
- Helps understand the power of Linear Regression with multiple features.

---

## 🧠 Key Concepts Implemented
- Linear Regression from scratch (no sklearn).  
- Cost Function (Mean Squared Error).  
- Gradient Descent Algorithm.  
- Weight (`w`) and Bias (`b`) optimization.  
- Data normalization and visualization with Matplotlib.  
- Evaluation with R² score and loss curve plotting.

---

## 📊 Single-variable vs Multi-variable

| Feature              | Single-variable                           | Multi-variable                                |
|-----------------------|-------------------------------------------|-----------------------------------------------|
| Input                | 1 feature                                | Multiple features                             |
| Visualization        | Easy (2D plot with line of best fit)      | Harder (can’t visualize easily in higher dims)|
| Example Projects     | House Price (size), Student Marks (hours) | House Price (size, rooms, location), Wine     |
| Use Case             | Simple predictions                       | Real-world complex datasets                   |

---

## ⚙️ Requirements
- Python 3.x  
- NumPy  
- Matplotlib  


# Linear Regression From Scratch

## Why This Repository?

Most tutorials use **Scikit-learn**, which hides the implementation details.  
Here, I implemented everything **from scratch** to strengthen my fundamentals in:

- **Machine Learning**  
- **Mathematics behind regression**  
- **Optimization techniques**  

This repository serves as a **learning resource** for beginners who want to deeply understand how **Linear Regression actually works**, rather than just using pre-built libraries.


## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by **Andrew Ng’s Machine Learning course** and other open-source learning materials.  
- Built as a **foundation step** before diving into advanced **Machine Learning (ML)** and **Deep Learning (DL)** projects.




