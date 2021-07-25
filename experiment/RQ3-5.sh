#!/bin/bash

cd ../adf_baseline

python ml_symbolic_generation.py --dataset=census --sens_param=1 --new_input=False --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=False
python ml_symbolic_generation.py --dataset=census --sens_param=1 --new_input=True --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=False
python ml_symbolic_generation.py --dataset=census --sens_param=1 --new_input=False --cluster_input=True --model_name=DecisionTreeClassifier --exp=RQ3 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=False

python ml_symbolic_generation.py --dataset=census --sens_param=8 --new_input=False --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=False
python ml_symbolic_generation.py --dataset=census --sens_param=8 --new_input=True --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=False
python ml_symbolic_generation.py --dataset=census --sens_param=8 --new_input=False --cluster_input=True --model_name=DecisionTreeClassifier --exp=RQ3 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=False

python ml_symbolic_generation.py --dataset=census --sens_param=9 --new_input=False --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=False
python ml_symbolic_generation.py --dataset=census --sens_param=9 --new_input=True --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=False
python ml_symbolic_generation.py --dataset=census --sens_param=9 --new_input=False --cluster_input=True --model_name=DecisionTreeClassifier --exp=RQ3 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=False

python ml_symbolic_generation.py --dataset=bank --sens_param=1 --new_input=False --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=False
python ml_symbolic_generation.py --dataset=bank --sens_param=1 --new_input=True --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=False
python ml_symbolic_generation.py --dataset=bank --sens_param=1 --new_input=False --cluster_input=True --model_name=DecisionTreeClassifier --exp=RQ3 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=False

python ml_symbolic_generation.py --dataset=compas --sens_param=1 --new_input=False --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=False
python ml_symbolic_generation.py --dataset=compas --sens_param=1 --new_input=True --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=False
python ml_symbolic_generation.py --dataset=compas --sens_param=1 --new_input=False --cluster_input=True --model_name=DecisionTreeClassifier --exp=RQ3 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=False

python ml_symbolic_generation.py --dataset=compas --sens_param=2 --new_input=False --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=False
python ml_symbolic_generation.py --dataset=compas --sens_param=2 --new_input=True --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=False
python ml_symbolic_generation.py --dataset=compas --sens_param=2 --new_input=False --cluster_input=True --model_name=DecisionTreeClassifier --exp=RQ3 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=False

python ml_symbolic_generation.py --dataset=compas --sens_param=3 --new_input=False --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=False
python ml_symbolic_generation.py --dataset=compas --sens_param=3 --new_input=True --cluster_input=False --model_name=DecisionTreeClassifier --exp=RQ3 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=False
python ml_symbolic_generation.py --dataset=compas --sens_param=3 --new_input=False --cluster_input=True --model_name=DecisionTreeClassifier --exp=RQ3 --sample_limit=1000 --max_iter=300 --cluster_num=4 --retraining=False