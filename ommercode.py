# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

def fit_gmm_log(data, max_components=6):

    log_data = np.log(data).reshape(-1, 1)

    bic_scores = []
    gmm_models = []

    for k in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(log_data)
        bic_scores.append(gmm.bic(log_data))
        gmm_models.append(gmm)

    optimal_index = np.argmin(bic_scores)
    optimal_gmm = gmm_models[optimal_index]

    print(f"Optimal number of components (K): {optimal_index + 1}")

    return optimal_gmm


def fit_xgboost(used_gas, cpu_time):


    model = XGBRegressor(objective='reg:squarederror', random_state=42)


    param_grid = {
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100, 150]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=0
    )

    grid_search.fit(used_gas.reshape(-1, 1), cpu_time)

    print("Best Parameters:", grid_search.best_params_)

    best_model = grid_search.best_estimator_

    return best_model


def sample_attributes(P, U, T, n_samples):

    log_gp_samples = P.sample(n_samples)[0]
    GP = np.exp(log_gp_samples).flatten()


    log_ug_samples = U.sample(n_samples)[0]
    UG = np.exp(log_ug_samples).flatten()

    GL = np.random.uniform(low=UG, high=15 * 10**6, size=n_samples)

    CT = T.predict(UG.reshape(-1, 1))

    return GP, UG, GL, CT


if __name__ == "__main__":


    np.random.seed(42)

    gas_price = np.random.lognormal(mean=3, sigma=0.5, size=5000)
    used_gas = np.random.lognormal(mean=10, sigma=0.4, size=5000)
    cpu_time = used_gas * 0.00001 + np.random.normal(0, 0.01, 5000)


    P = fit_gmm_log(gas_price)

    U = fit_gmm_log(used_gas)

    T = fit_xgboost(used_gas, cpu_time)

    GP_sample, UG_sample, GL_sample, CT_sample = sample_attributes(P, U, T, n_samples=1000)

    print("\nSampled Data Shapes:")
    print("Gas Price:", GP_sample.shape)
    print("Used Gas:", UG_sample.shape)
    print("Gas Limit:", GL_sample.shape)
    print("CPU Time:", CT_sample.shape)
