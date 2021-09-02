# Getting Started

## Environment
Python 3.6, Tensorflow 1.11, and scikit-learn 0.20 are suggested. The packages numpy, pandas, and joblib are also required. Please refer to `requirements.txt`.

## A Demo

We provide a demo. Please run `python example.py` in the corresponding file directory.

|Dataset|Index|Protected Attribute|Link|
|----|----|----|----|
|Census|1|age|https://archive.ics.uci.edu/ml/datasets/adult|
||8|race||
||9|sex||
|Compas|1|sex|https://github.com/propublica/compas-analysis|
||2|age||
||3|race||
|Bank|1|age|https://archive.ics.uci.edu/ml/datasets/bank+marketing|

## Parameter Description in the Demo
### iandd
1. `(dataset,sensitive_param)`: The name of the dataset and the index of the protected attribute. Optional parameters: ` ('census', 1)`, `('census', 8)`, ` ('census', 9)`, `('compas', 1)`, `('compas', 2)`, `('compas', 3)`, `('bank', 1)`.
2. `max_iter`: Number of iterations of training.
3. `sample_limit`: Initial seeds limit.
4. `model_name`: Model name. Optional parameters: `'LogisticRegression'`, `'DecisionTreeClassifier'`, `'MLPClassifier'`, `'SVC'`, `'NN'`.

### aequitas
1. `(dataset,sensitive_param)`: The name of the dataset and the index of the protected attributes. Optional parameters: ` ('census', 1)`, `('census', 8)`, ` ('census', 9)`, `('compas', 1)`, `('compas', 2)`, `('compas', 3)`, `('bank', 1)`.
2. `max_global`: Global limit.
3. `max_loacal`: Local limit.
4. `max_iter`: Number of iterations of training.
5. `step_size`: Step size for perturbation.
6. `model_name`: Model name. Optional parameters: `'LogisticRegression'`, `'DecisionTreeClassifier'`, `'MLPClassifier'`, `'SVC'`, `'NN'`.

### retraining_testing
1. `(dataset,sensitive_param)`: The name of the dataset and the ordinal number of the sensitive attributes. Optional parameters: ` ('census', 1)`, `('census', 8)`, ` ('census', 9)`, `('compas', 1)`, `('compas', 2)`, `('compas', 3)`, `('bank', 1)`.
2. `max_iter`: Number of iterations of training.
3. `approach_name`: Approach name. Optional parameters: `'AEQ'`, `'SG'`, `'ADF'`, `'EIDIG'`.
4. `model_name`: Model name. Optional parameters: `'LogisticRegression'`, `'DecisionTreeClassifier'`, `'MLPClassifier'`, `'SVC'`, `'NN'`.

## Citation
Please cite our paper: tba
