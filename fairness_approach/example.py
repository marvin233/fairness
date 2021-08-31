# AEQ., LogisticRegression, census, age

# initial idi
from iandd import iandd
iandd(dataset='census', sensitive_param=1, max_iter=30, sample_limit=100, model_name='LogisticRegression')



# idi number
from AEQ import aequitas
aequitas(dataset='census', sensitive_param=1, max_global=100, max_local=100, max_iter=30, step_size=1.0, model_name='LogisticRegression')

# original
# global 2
# local 92
# disc_inputs 92
# ==================================
# w_I
# global 100
# local 5624
# disc_inputs 5624
# ==================================
# w_I_D
# global 95
# local 6697
# disc_inputs 6697
# ==================================


# retraining & testing
from retraining_testing import retraining_testing
retraining_testing(dataset='census', sensitive_param=1, max_iter=30, approach_name='AEQ', model_name='LogisticRegression')

# F1-Score: 0.8808456420178378
# ==================================
# original
# Retraining F1-Score: 0.8811572279474331
# 6529/6789
# ==================================
# w_I_D
# Retraining F1-Score: 0.8823173287384157
# 3570/6789
# ==================================