import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(file_path, feature_columns, target_column):
    data = pd.read_csv(file_path)
    data = data.dropna()

    X = data[feature_columns]
    Y = data[target_column]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, Y


def manual_gpr_tuning(X_train, y_train, kernel_list, alphas):
    best_gp = None
    best_mse = np.inf
    best_params = {}

    for kernel in kernel_list:
        for alpha in alphas:
            gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha)
            gp.fit(X_train, y_train)

            y_train_pred, _ = gp.predict(X_train, return_std=True)
            mse_train = mean_squared_error(y_train, y_train_pred)

            if mse_train < best_mse:
                best_gp = gp
                best_mse = mse_train
                best_params = {
                    'kernel': kernel,
                    'alpha': alpha
                }

    print(f"Best MSE: {best_mse}, Best params: {best_params}")
    return best_gp, best_params


def train_and_evaluate_model(X_train, X_test, y_train, y_test, kernel_list, alphas):
    best_gp, best_params = manual_gpr_tuning(X_train, y_train, kernel_list, alphas)
    y_pred, y_std = best_gp.predict(X_test, return_std=True)
    mse_test = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on Test Data: {mse_test}")

    return y_pred, y_std, mse_test


def display_results(X_test, y_pred, y_std, original_data):
    data_test = original_data.iloc[X_test.index]
    data_test['PredictedVulnerability'] = y_pred
    data_test['Uncertainty'] = y_std

    print(data_test[['DestinationPort', 'PredictedVulnerability', 'Uncertainty']])
    return data_test


def main():
    file_path = '/Users/manzifabriceniyigaba/Desktop/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'
    feature_columns = ['DestinationPort', 'FlowDuration', 'TotalFwdPackets', 'TotalBwdPackets']
    target_column = 'Vulnerability'

    X_scaled, Y = load_and_preprocess_data(file_path, feature_columns, target_column)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)

    # kernels and alpha values for tuning
    kernel_list = [
        C(1.0) * RBF(length_scale=1.0),
        C(1.0) * Matern(length_scale=1.0, nu=1.5)
    ]
    alphas = [1e-10, 1e-5, 1e-2]
    y_pred, y_std, mse_test = train_and_evaluate_model(X_train, X_test, y_train, y_test, kernel_list, alphas)

    original_data = pd.read_csv(file_path)
    display_results(X_test, y_pred, y_std, original_data)

if __name__ == "__main__":
    main()

