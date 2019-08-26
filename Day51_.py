'''
EconML from Microsoft
python econometrics library with ML automation for causal inference
'''

# Double ML
from econml.dml import DMLCateEstimator
from sklearn.linear_model import LassoCV

est = DMLCateEstimator(model_y=LassoCV(), model_t=LassoCV())
est.fit(Y, T, X, W) # W -> high-dimensional confounders, X -> features
treatment_effects = est.effect(X_test)

# Orthogonal Random Forest

from econml.ortho_forest import ContinuousTreatmentOrthoForest
# Use defaults
est = ContinuousTreatmentOrthoForest()
# Or specify hyperparameters
est = ContinuousTreatmentOrthoForest(n_trees=500, min_leaf_size=10, max_depth=10, 
                                     subsample_ratio=0.7, lambda_reg=0.01,
                                     model_T=LassoCV(cv=3), model_Y=LassoCV(cv=3)
                                     )
est.fit(Y, T, X, W)
treatment_effects = est.effect(X_test)