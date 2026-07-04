# SCO2_Nusselt_Number_Prediction
Machine Learning Models for Predicting local Nusselt Number of CO2 in Supercritical phase Using Experimental Thermodynamic and Flow Characteristics Data, 
5400 Data Points Extracted from Experimental Published Researches,
Structured with 8 Parameters as known Variables Including 8 Columns in Dataset,
Multi Layer Perceptron Model,
Practiced both in Python and MATLAB Softwares.

================ RESULTS of MLP Alghorithms ===========================

R2 Score = 0.7239

RMSE = 10.3981

MAE = 6.9373

MAPE (%) = 32.28

These results represent the baseline model. Future work includes dataset refinement, hyperparameter optimization, and comparison with XGBoost

=========================================================================


================ RESULTS of XGBoost Performin alghorithms ===============

R2 Score = 0.9849

RMSE     = 2.4340

MAE      = 0.9123

MAPE     = 4.09%

=========================================================================

MLP (Multilayer Perceptron)
R² Score: 0.7239
RMSE: 10.3981
MAE: 6.9373
MAPE: 32.28%

The MLP successfully learned the nonlinear relationship between the eight input parameters and the local Nusselt number. However, its prediction accuracy is moderate, with relatively high prediction errors, indicating that the network could not fully capture the complex thermophysical behavior of supercritical CO₂.

XGBoost (Extreme Gradient Boosting)
R² Score: 0.9849
RMSE: 2.4340
MAE: 0.9123
MAPE: 4.09%

The XGBoost model demonstrated excellent predictive capability, accurately capturing the nonlinear relationship between the operating parameters and the local Nusselt number. Compared with the MLP, it achieved substantially lower prediction errors and a much higher coefficient of determination, indicating superior generalization on the test dataset.

Overall Comparison

Both models were trained using the same experimental dataset and evaluated using the same train-test split and performance metrics. Although the MLP provided reasonable predictions, XGBoost consistently outperformed it in every evaluation metric. The coefficient of determination increased from 0.7239 to 0.9849, while the RMSE, MAE, and MAPE were reduced from 10.40, 6.94, and 32.28% to 2.43, 0.91, and 4.09%, respectively. These results indicate that XGBoost is considerably more effective than the current MLP model for predicting the local Nusselt number of supercritical CO₂ under the investigated operating conditions.
