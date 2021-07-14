import sys
sys.path.append("../")
import numpy as np

"""
Pre-process the raw data of Credit,
convert the text to numerical data.
"""

# list all the values of enumerate features


RecSupervisionLevel = []
DisplayText = []
RawScore = []
AssessmentType = []

data = []
s = ''
for c in [chr(i) for i in range(97, 97+11)]:
    s = s + c + ','
print(s)
with open("../datasets/executions_raw/execution_database.csv", "r") as ins:
    for line in ins:
        line = line.strip()
        features = line.split(',')
        temp_features = []
        temp_features.append(int(int(features[2])/10))# age
        temp_features.append(Sex.index(features[3]))# sex
        temp_features.append(Race.index(features[4]))# race
        temp_features.append(State.index(features[6]))# state
        temp_features.append(Region.index(features[7]))# region
        temp_features.append(Method.index(features[8]))# method
        temp_features.append(Juvenile.index(features[9]))# juvenile
        temp_features.append(Federal.index(features[10]))# federal
        temp_features.append(Volunteer.index(features[11]))# volunteer
        temp_features.append(Foreign_National.index(features[12]))# Foreign National
        temp_features.append(1)
        data.append(temp_features)
data = np.asarray(data)

np.savetxt("../datasets/execution", data, fmt="%d",delimiter=",")