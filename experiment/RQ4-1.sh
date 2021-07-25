#!/bin/bash

cd ../adf_baseline

python ml_aequitas.py --dataset=census --sens_param=1 --model_name=LogisticRegression --exp=RQ4 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=True --new_input=False --cluster_input=False
python ml_aequitas.py --dataset=census --sens_param=8 --model_name=LogisticRegression --exp=RQ4 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=True --new_input=False --cluster_input=False
python ml_aequitas.py --dataset=census --sens_param=9 --model_name=LogisticRegression --exp=RQ4 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=True --new_input=False --cluster_input=False
python ml_aequitas.py --dataset=bank --sens_param=1 --model_name=LogisticRegression --exp=RQ4 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=True --new_input=False --cluster_input=False
python ml_aequitas.py --dataset=compas --sens_param=1 --model_name=LogisticRegression --exp=RQ4 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=True --new_input=False --cluster_input=False
python ml_aequitas.py --dataset=compas --sens_param=2 --model_name=LogisticRegression --exp=RQ4 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=True --new_input=False --cluster_input=False
python ml_aequitas.py --dataset=compas --sens_param=3 --model_name=LogisticRegression --exp=RQ4 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=True --new_input=False --cluster_input=False

python ml_aequitas.py --dataset=census --sens_param=1 --model_name=DecisionTreeClassifier --exp=RQ4 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=True --new_input=False --cluster_input=False
python ml_aequitas.py --dataset=census --sens_param=8 --model_name=DecisionTreeClassifier --exp=RQ4 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=True --new_input=False --cluster_input=False
python ml_aequitas.py --dataset=census --sens_param=9 --model_name=DecisionTreeClassifier --exp=RQ4 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=True --new_input=False --cluster_input=False
python ml_aequitas.py --dataset=bank --sens_param=1 --model_name=DecisionTreeClassifier --exp=RQ4 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=True --new_input=False --cluster_input=False
python ml_aequitas.py --dataset=compas --sens_param=1 --model_name=DecisionTreeClassifier --exp=RQ4 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=True --new_input=False --cluster_input=False
python ml_aequitas.py --dataset=compas --sens_param=2 --model_name=DecisionTreeClassifier --exp=RQ4 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=True --new_input=False --cluster_input=False
python ml_aequitas.py --dataset=compas --sens_param=3 --model_name=DecisionTreeClassifier --exp=RQ4 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=True --new_input=False --cluster_input=False

python ml_aequitas.py --dataset=census --sens_param=1 --model_name=MLPClassifier --exp=RQ4 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=True --new_input=False --cluster_input=False
python ml_aequitas.py --dataset=census --sens_param=8 --model_name=MLPClassifier --exp=RQ4 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=True --new_input=False --cluster_input=False
python ml_aequitas.py --dataset=census --sens_param=9 --model_name=MLPClassifier --exp=RQ4 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=True --new_input=False --cluster_input=False
python ml_aequitas.py --dataset=bank --sens_param=1 --model_name=MLPClassifier --exp=RQ4 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=True --new_input=False --cluster_input=False
python ml_aequitas.py --dataset=compas --sens_param=1 --model_name=MLPClassifier --exp=RQ4 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=True --new_input=False --cluster_input=False
python ml_aequitas.py --dataset=compas --sens_param=2 --model_name=MLPClassifier --exp=RQ4 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=True --new_input=False --cluster_input=False
python ml_aequitas.py --dataset=compas --sens_param=3 --model_name=MLPClassifier --exp=RQ4 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=True --new_input=False --cluster_input=False