# load dataset
from pycaret.datasets import get_data
insurance = get_data('insurance')

# init environment
from pycaret.regression import *
r1 = setup(insurance, target = 'charges', session_id = 123,
           normalize = True,
           polynomial_features = True,
           feature_interaction=True,
           bin_numeric_features= ['age', 'bmi'])

# train a model
lr = create_model('lr')

# save pipeline/model
save_model(lr, model_name = 'deployment_20241101')
