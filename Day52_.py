'''
Day 52
Uber CausalML

'''

from causalml.inference import LRSRegressor
from causalml.inference import XGBTRegressor, MLPTRegressor
from causalml.inference import BaseXRegressor
from causalml.dataset import synthetic_data

y, X, treatment, _ = synthetic_data(mode=1, n=1000, p=5, sigma=1.0)

lr = LRSRegressor()
te, lb, ub = lr.estimate_ate(X, treatment, y)
logger.info('Average Treatment Effect (Linear Regression): {:.2f} ({:.2f}, {:.2f})'.format(te, lb, ub))

xg = XGBTRegressor(random_state=42)
te, lb, ub = xg.estimate_ate(X, treatment, y)
logger.info('Average Treatment Effect (XGBoost): {:.2f} ({:.2f}, {:.2f})'.format(te, lb, ub))

nn = MLPTRegressor(hidden_layer_sizes=(10, 10),
                 learning_rate_init=.1,
                 early_stopping=True,
                 random_state=42)
te, lb, ub = nn.estimate_ate(X, treatment, y)
logger.info('Average Treatment Effect (Neural Network (MLP)): {:.2f} ({:.2f}, {:.2f})'.format(te, lb, ub))

xl = BaseXRegressor(learner=XGBRegressor(random_state=42))
te, lb, ub = xl.estimate_ate(X, p, treatment, y)
logger.info('Average Treatment Effect (XGBoost): {:.2f} ({:.2f}, {:.2f})'.format(te, lb, ub))