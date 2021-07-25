#!/bin/bash

cd ../adf_baseline

python ml_aequitas.py --dataset=census --sens_param=1 --new_input=False --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=False
python ml_aequitas.py --dataset=census --sens_param=1 --new_input=True --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=False
python ml_aequitas.py --dataset=census --sens_param=1 --new_input=False --cluster_input=True --model_name=DecisionTreeClassifier --exp=RQ3 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=False

python ml_aequitas.py --dataset=census --sens_param=8 --new_input=False --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=False
python ml_aequitas.py --dataset=census --sens_param=8 --new_input=True --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=False
python ml_aequitas.py --dataset=census --sens_param=8 --new_input=False --cluster_input=True --model_name=DecisionTreeClassifier --exp=RQ3 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=False

python ml_aequitas.py --dataset=census --sens_param=9 --new_input=False --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=False
python ml_aequitas.py --dataset=census --sens_param=9 --new_input=True --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=False
python ml_aequitas.py --dataset=census --sens_param=9 --new_input=False --cluster_input=True --model_name=DecisionTreeClassifier --exp=RQ3 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=False

python ml_aequitas.py --dataset=bank --sens_param=1 --new_input=False --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=False
python ml_aequitas.py --dataset=bank --sens_param=1 --new_input=True --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=False
python ml_aequitas.py --dataset=bank --sens_param=1 --new_input=False --cluster_input=True --model_name=DecisionTreeClassifier --exp=RQ3 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=False

python ml_aequitas.py --dataset=compas --sens_param=1 --new_input=False --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=False
python ml_aequitas.py --dataset=compas --sens_param=1 --new_input=True --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=False
python ml_aequitas.py --dataset=compas --sens_param=1 --new_input=False --cluster_input=True --model_name=DecisionTreeClassifier --exp=RQ3 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=False

python ml_aequitas.py --dataset=compas --sens_param=2 --new_input=False --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=False
python ml_aequitas.py --dataset=compas --sens_param=2 --new_input=True --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=False
python ml_aequitas.py --dataset=compas --sens_param=2 --new_input=False --cluster_input=True --model_name=DecisionTreeClassifier --exp=RQ3 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=False

python ml_aequitas.py --dataset=compas --sens_param=3 --new_input=False --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=False
python ml_aequitas.py --dataset=compas --sens_param=3 --new_input=True --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=False
python ml_aequitas.py --dataset=compas --sens_param=3 --new_input=False --cluster_input=True --model_name=DecisionTreeClassifier --exp=RQ3 --max_global=1000 --max_local=1000 --max_iter=300 --step_size=1.0 --retraining=False