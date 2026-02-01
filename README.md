# Remaining-Useful-Life-Prediction-for-Turbofan-Engines-NASA-CMAPSS-

*OVERVIEW*

This project addresses the prediction of Remaining Useful Life (RUL) for aircraft turbofan engines using multivariate sensor data from the NASA CMAPSS dataset. Each engine is modeled as an independent temporal sequence of operating cycles up to failure. The main objective is to assess how far classical tabular machine learning models can go in capturing degradation patterns under a realistic evaluation protocol.
Rather than optimizing a single model blindly, the project emphasizes methodological rigor: consistent train/validation splits by engine, careful handling of temporal information, and interpretation of performance limits

*STRUCTURE*

├── data/
│   ├── raw/cleaned files             
│            
│
├── src/
│   ├── data_cleaning.py      # Data cleaning and formatting
│   ├── feature_eng.py        # Rolling feature engineering
│   └── train.py              # Training, validation, and test evaluation
│
├── README.md

*DATASET*

NASA CMAPSS – Turbofan Engine Degradation Simulation
- Each unit_id corresponds to a physical engine
- Each row represents one operational cycle with multiple sensor readings
- Training data includes RUL labels
- Test data does not include RUL; true values are provided separately (RUL_FD001.txt)

*METHODOLOGY*

The project follows a structured, progressive approach:
  Baselines and Linear Models
    - Simple baselines (mean predictor) and linear models (Linear Regression, Ridge, Lasso) are used to establish a reference level of performance and identify the practical limit of linear approaches.

Temporal Feature Engineering
  Short-term memory is introduced through rolling statistics computed per engine:
    - rolling mean (state estimation)
    - rolling slope (local degradation trend)
    - rolling standard deviation (local instability)

Several non-linear models are evaluated and tuned under the same protocol:
    - Random Forest
    - ExtraTrees
    - HistGradientBoosting
    - XGBoost
    - LightGBM
    
(Hyperparameter search is deliberately constrained to avoid overfitting and excessive compute)

Evaluation Protocol
    - Metric: Mean Absolute Error (MAE), expressed in number of cycles
    - Validation split: by engine (fixed subset of units)
    - Final test evaluation: one prediction per engine, using the last available cycle, compared against official CMAPSS RUL targets

*RESULTS*

Baseline (mean predictor)
    - MAE ≈ 56 cycles
    - Confirms that naive predictors are not informative.
    
Linear models (no feature engineering)
    - Linear Regression / Ridge / Lasso: MAE ≈ 25 
    - Large performance jump, but clear saturation of linear models.

Linear models + deltas
    - Marginal improvement only (MAE ≈ 25.0)
    - Indicates limited additional signal from 1-step differences.

Linear models + rolling mean / slope / std
    - No improvement (sometimes worse than raw features).
    - Confirms representational limits of linear models.

(Although rolling mean, slope, and standard deviation did not improve linear models, they were retained because they 
provide a physically meaningful temporal representation that non-linear models can exploit more effectively)

Random Forest
    - Validation MAE ≈ 22.5 after tuning.
    - Non-linear interactions provide additional but limited gains.

ExtraTrees (best model)
    - Validation MAE ≈ 21.9
    - Best-performing model among all tested approaches.
    - Key characteristics: moderately deep trees, feature subsampling, minimum leaf size regularization.

Boosting models (HGB, XGBoost, LightGBM)
    - Competitive but inferior to ExtraTrees (MAE ≈ 23–24).
    - No significant advantage in this setup.

Final test result (FD001)
    - ExtraTrees evaluated on official test set (last cycle per engine).
    - Test MAE ≈ 18.5 cycles.

*CONCLUSION*

This study shows that, for the NASA CMAPSS FD001 dataset, most of the easily accessible predictive signal can be captured by simple linear models, with non-linear tabular models providing only moderate additional gains. Despite a slight degradation in linear performance, rolling temporal features (mean, slope, and standard deviation) were maintained due to their physical interpretability and their compatibility with tree-based models. ExtraTrees achieved the best overall performance, reaching a validation MAE of approximately 21.9 cycles and a final test MAE of about 18.5 cycles. These results indicate a clear performance plateau for tabular approaches with short-term temporal features, suggesting that further improvements would likely require richer temporal modeling or alternative problem formulations.


AUTHOR Rodrigo Driemeier dos Santos EESC – University of São Paulo (USP), São Carlos, Brazil — Mechatronics Engineering École Centrale de Lille, France — Generalist Engineering
📧 rodrigo.driemeier@centrale.centralelille.fr
📧 rodrigodriemeier@usp.br
🔗 https://www.linkedin.com/in/rodrigo-driemeier-dos-santos-a7698633b/
Thanks for checking out the project :)
