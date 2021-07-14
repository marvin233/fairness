import sys
sys.path.append("../")
import numpy as np

"""
Pre-process the raw data of Credit,
convert the text to numerical data.
"""

# list all the values of enumerate features
checking_status = ['A11', 'A12', 'A13', 'A14']
duration = []
credit_history = ['A30', 'A31', 'A32', 'A33', 'A34']
purpose = ['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', 'A410']
credit_amount = []
savings_status = ['A61', 'A62', 'A63', 'A64', 'A65']
employment = ['A71', 'A72', 'A73', 'A74', 'A75']
installment_commitment = []
sex = ['A91', 'A92', 'A93', 'A94', 'A95']
other_parties = ['A101', 'A102', 'A103']
residence = []
property_magnitude = ['A121', 'A122', 'A123', 'A124']
age = []
other_payment_plans = ['A141', 'A142', 'A143']
housing = ['A151', 'A152', 'A153']
existing_credits = []
job = ['A171', 'A172', 'A173', 'A174']
num_dependents = []
own_telephone = ['A191', 'A192']
foreign_worker = ['A201', 'A202']
yy = ['1', '2']

data = []
s = ''
for c in [chr(i) for i in range(97, 97+21)]:
    s = s + c + ','
print(s)
count = 0
with open("../datasets/credit_raw/german.data", "r") as ins:
    for line in ins:
        line = line.strip()
        features = line.split(' ')
        features[0] = checking_status.index(features[0])
        features[1] = int(features[1])
        features[2] = credit_history.index(features[2])
        features[3] = purpose.index(features[3])
        features[4] = np.clip(int(features[4])/100, 1, 200)
        # features[4] = int(features[4])
        features[5] = savings_status.index(features[5])
        features[6] = employment.index(features[6])
        features[7] = int(features[7])
        if sex.index(features[8]) in [0, 2, 3]:
            features[8] = 0
        else:
            features[8] = 1
        features[9] = other_parties.index(features[9])
        features[10] = int(features[10])
        features[11] = property_magnitude.index(features[11])
        features[12] = np.clip(int(features[12])/10, 1, 8)
        features[13] = other_payment_plans.index(features[13])
        features[14] = housing.index(features[14])
        features[15] = int(features[15])
        features[16] = job.index(features[16])
        features[17] = int(features[17])
        features[18] = own_telephone.index(features[18])
        features[19] = foreign_worker.index(features[19])
        if int(features[20]) == 1:
            features[20] = 1
            count += 1
            if count > 300:
                continue
        else:
            features[20] = 0
        data.append(features)
        print(features[4], features[12])

data = np.asarray(data)
np.savetxt("../datasets/credit", data, fmt="%d",delimiter=",")