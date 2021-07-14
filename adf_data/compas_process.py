import sys
sys.path.append("../")
import numpy as np

"""
Pre-process the raw data of Credit,
convert the text to numerical data.
"""

# list all the values of enumerate features
Agency_Text = ['PRETRIAL', 'Probation', 'DRRD', 'Broward County']
Sex_Code_Text = ['Male', 'Female']
Ethnic_Code_Text = ['Caucasian', 'African-American', 'Hispanic', 'Other', 'Asian', 'African-Am', 'Native American', 'Oriental', 'Arabic']
ScaleSet = ['Risk and Prescreen', 'All Scales']
Language = ['English', 'Spanish']
LegalStatus = ['Pretrial', 'Post Sentence', 'Conditional Release', 'Other', 'Probation Violator', 'Parole Violator', 'Deferred Sentencing']
CustodyStatus = ['Jail Inmate', 'Probation', 'Pretrial Defendant', 'Residential Program', 'Prison Inmate', 'Parole']
MaritalStatus = ['Single', 'Married', 'Significant Other', 'Divorced', 'Separated', 'Widowed', 'Unknown']
RecSupervisionLevel = ['1', '2', '3', '4']
DisplayText = ['Risk of Violence', 'Risk of Recidivism', 'Risk of Failure to Appear']
RawScore = []
AssessmentType = ['New', 'Copy']
data = []
s = ''
for c in [chr(i) for i in range(97, 97+13)]:
    s = s + c + ','
print(s)
with open("../datasets/compas_raw/compas-scores-raw.csv", "r") as ins:
    for line in ins:
        line = line.strip()
        features = line.split(',')
        if 'N/A' in features:
            continue
        features[0] = Agency_Text.index(features[0])
        features[1] = Sex_Code_Text.index(features[1])
        features[2] = Ethnic_Code_Text.index(features[2])
        features[3] = ScaleSet.index(features[3])
        features[4] = Language.index(features[4])
        features[5] = LegalStatus.index(features[5])
        features[6] = CustodyStatus.index(features[6])
        features[7] = MaritalStatus.index(features[7])
        features[8] = RecSupervisionLevel.index(features[8])
        features[9] = DisplayText.index(features[9])
        features[10] = int(float(features[10])+ 5)
        features[11] = AssessmentType.index(features[11])
        if int(features[12]) <= 5:
            features[12] = 0
        else:
            features[12] = 1
        data.append(features)
data = np.asarray(data)
print(len(data))
print(np.sum(data[:,12]))
np.savetxt("../datasets/compas", data, fmt="%d",delimiter=",")