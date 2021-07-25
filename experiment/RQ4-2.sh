#!/bin/bash

cd ../adf_baseline

python ml_symbolic_generation.py --dataset=census --sens_param=1 --model_name=LogisticRegression --exp=RQ4 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=True --new_input=False --cluster_input=False
python ml_symbolic_generation.py --dataset=census --sens_param=8 --model_name=LogisticRegression --exp=RQ4 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=True --new_input=False --cluster_input=False
python ml_symbolic_generation.py --dataset=census --sens_param=9 --model_name=LogisticRegression --exp=RQ4 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=True --new_input=False --cluster_input=False
python ml_symbolic_generation.py --dataset=bank --sens_param=1 --model_name=LogisticRegression --exp=RQ4 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=True --new_input=False --cluster_input=False
python ml_symbolic_generation.py --dataset=compas --sens_param=1 --model_name=LogisticRegression --exp=RQ4 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=True --new_input=False --cluster_input=False
python ml_symbolic_generation.py --dataset=compas --sens_param=2 --model_name=LogisticRegression --exp=RQ4 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=True --new_input=False --cluster_input=False
python ml_symbolic_generation.py --dataset=compas --sens_param=3 --model_name=LogisticRegression --exp=RQ4 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=True --new_input=False --cluster_input=False

python ml_symbolic_generation.py --dataset=census --sens_param=1 --model_name=DecisionTreeClassifier --exp=RQ4 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=True --new_input=False --cluster_input=False
python ml_symbolic_generation.py --dataset=census --sens_param=8 --model_name=DecisionTreeClassifier --exp=RQ4 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=True --new_input=False --cluster_input=False
python ml_symbolic_generation.py --dataset=census --sens_param=9 --model_name=DecisionTreeClassifier --exp=RQ4 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=True --new_input=False --cluster_input=False
python ml_symbolic_generation.py --dataset=bank --sens_param=1 --model_name=DecisionTreeClassifier --exp=RQ4 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=True --new_input=False --cluster_input=False
python ml_symbolic_generation.py --dataset=compas --sens_param=1 --model_name=DecisionTreeClassifier --exp=RQ4 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=True --new_input=False --cluster_input=False
python ml_symbolic_generation.py --dataset=compas --sens_param=2 --model_name=DecisionTreeClassifier --exp=RQ4 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=True --new_input=False --cluster_input=False
python ml_symbolic_generation.py --dataset=compas --sens_param=3 --model_name=DecisionTreeClassifier --exp=RQ4 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=True --new_input=False --cluster_input=False

python ml_symbolic_generation.py --dataset=census --sens_param=1 --model_name=MLPClassifier --exp=RQ4 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=True --new_input=False --cluster_input=False
python ml_symbolic_generation.py --dataset=census --sens_param=8 --model_name=MLPClassifier --exp=RQ4 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=True --new_input=False --cluster_input=False
python ml_symbolic_generation.py --dataset=census --sens_param=9 --model_name=MLPClassifier --exp=RQ4 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=True --new_input=False --cluster_input=False
python ml_symbolic_generation.py --dataset=bank --sens_param=1 --model_name=MLPClassifier --exp=RQ4 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=True --new_input=False --cluster_input=False
python ml_symbolic_generation.py --dataset=compas --sens_param=1 --model_name=MLPClassifier --exp=RQ4 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=True --new_input=False --cluster_input=False
python ml_symbolic_generation.py --dataset=compas --sens_param=2 --model_name=MLPClassifier --exp=RQ4 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=True --new_input=False --cluster_input=False
python ml_symbolic_generation.py --dataset=compas --sens_param=3 --model_name=MLPClassifier --exp=RQ4 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=True --new_input=False --cluster_input=False
