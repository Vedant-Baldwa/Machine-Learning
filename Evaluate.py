import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_regression(y_true, y_pred):
    # Error metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)

    # Print metrics
    print(f"📏 Evaluation Metrics:\n"
          f"MAE  = {mae:.2f}\n"
          f"MSE  = {mse:.2f}\n"
          f"RMSE = {rmse:.2f}\n"
          f"R²   = {r2:.2f}")

    # Plot layout
    plt.figure(figsize=(18, 5))

    # 1. Actual vs Predicted
    plt.subplot(1, 3, 1)
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.grid(True)

    # 2. Residual distribution
    residuals = y_true - y_pred
    plt.subplot(1, 3, 2)
    sns.histplot(residuals, kde=True, bins=30, color='purple')
    plt.title('Residual Distribution')
    plt.xlabel('Residuals')
    plt.grid(True)

    # 3. Residuals vs Predictions
    plt.subplot(1, 3, 3)
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
