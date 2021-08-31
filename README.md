# Getting Started

## Environment

Python3.6 + Tensorflow1.11 + scikit-learn0.20 are recommended. The following packages are also required: numpy, pandas, joblib. The specific parameters are in the file 'requirements.txt'.

## A Small Example

We provide a small example. In the corresponding file directory, enter in the terminal 'python example.py'.

## Parameter Description in Example
### iandd
1. (dataset,sensitive_param): The name of the dataset and the ordinal number of the sensitive attributes. Optional parameters: ` ('census', 1)`, `('census', 8)`, ` ('census', 9)`, `('compas', 1)`, `('compas', 2)`, `('compas', 3)`, `('bank', 1)`.
2. max_iter: Number of iterations of training.
3. sample_limit: Initial seeds limit.
4. model_name: Model name. Optional parameters: `'LogisticRegression'`, `'DecisionTreeClassifier'`, `'MLPClassifier'`, `'SVC'`, `'NN'`.

### aequitas
1. (dataset,sensitive_param): The name of the dataset and the ordinal number of the sensitive attributes. Optional parameters: ` ('census', 1)`, `('census', 8)`, ` ('census', 9)`, `('compas', 1)`, `('compas', 2)`, `('compas', 3)`, `('bank', 1)`.
2. max_global: Global limit.
3. max_loacal: Local limit.
4. max_iter: Number of iterations of training.
5. step_size: Step size for perturbation.
6. model_name: Model name. Optional parameters: `'LogisticRegression'`, `'DecisionTreeClassifier'`, `'MLPClassifier'`, `'SVC'`, `'NN'`.

### retraining_testing
1. (dataset,sensitive_param): The name of the dataset and the ordinal number of the sensitive attributes. Optional parameters: ` ('census', 1)`, `('census', 8)`, ` ('census', 9)`, `('compas', 1)`, `('compas', 2)`, `('compas', 3)`, `('bank', 1)`.
2. max_iter: Number of iterations of training.
3. approach_name: Approach name. Optional parameters: `'AEQ'`, `'SG'`, `'ADF'`, `'EIDIG'`.
4. model_name: Model name. Optional parameters: `'LogisticRegression'`, `'DecisionTreeClassifier'`, `'MLPClassifier'`, `'SVC'`, `'NN'`.