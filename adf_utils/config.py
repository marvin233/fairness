class census:
    """
    Configuration of dataset Census Income
    """

    # the size of total features
    params = 13

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([1, 9])
    input_bounds.append([0, 7])
    input_bounds.append([0, 39]) #69 for THEMIS
    input_bounds.append([0, 15])
    input_bounds.append([0, 6])
    input_bounds.append([0, 13])
    input_bounds.append([0, 5])
    input_bounds.append([0, 4])
    input_bounds.append([0, 1])
    input_bounds.append([0, 99])
    input_bounds.append([0, 39])
    input_bounds.append([0, 99])
    input_bounds.append([0, 39])

    # the name of each feature
    feature_name = ["age",
                    "workclass",
                    "fnlwgt",
                    "education",
                    "marital_status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "capital_gain",
                    "capital_loss",
                    "hours_per_week",
                    "native_country"]

    # the name of each class
    class_name = ["low", "high"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


class credit:
    """
    Configuration of dataset German Credit
    """

    # the size of total features
    params = 20

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([0, 3]) #1
    input_bounds.append([1, 80]) #2
    input_bounds.append([0, 4]) #3
    input_bounds.append([0, 10]) #4
    input_bounds.append([1, 200]) #5
    input_bounds.append([0, 4]) #6
    input_bounds.append([0, 4]) #7
    input_bounds.append([1, 4]) #8
    input_bounds.append([0, 1]) #9
    input_bounds.append([0, 2])#10
    input_bounds.append([1, 4])#11
    input_bounds.append([0, 3])#12
    input_bounds.append([1, 8])#13
    input_bounds.append([0, 2])#14
    input_bounds.append([0, 2])#15
    input_bounds.append([1, 4])#16
    input_bounds.append([0, 3])#17
    input_bounds.append([1, 2])#18
    input_bounds.append([0, 1])#19
    input_bounds.append([0, 1])#20

    # the name of each feature
    feature_name = ["checking_status",
                    "duration",
                    "credit_history",
                    "purpose",
                    "credit_amount",
                    "savings_status",
                    "employment",
                    "installment_commitment",
                    "sex",
                    "other_parties",
                    "residence",
                    "property_magnitude",
                    "age",
                    "other_payment_plans",
                    "housing",
                    "existing_credits",
                    "job",
                    "num_dependents",
                    "own_telephone",
                    "foreign_worker"]

    # the name of each class
    class_name = ["bad", "good"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17, 18, 19]


class bank:
    """
    Configuration of dataset Bank Marketing
    """

    # the size of total features
    params = 16

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([1, 9])
    input_bounds.append([0, 11])
    input_bounds.append([0, 2])
    input_bounds.append([0, 3])
    input_bounds.append([0, 1])
    input_bounds.append([-20, 179])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 2])
    input_bounds.append([1, 31])
    input_bounds.append([0, 11])
    input_bounds.append([0, 99])
    input_bounds.append([1, 63])
    input_bounds.append([-1, 39])
    input_bounds.append([0, 1])
    input_bounds.append([0, 3])

    # the name of each feature
    feature_name = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"]

    # the name of each class
    class_name = ["no", "yes"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


class execution:
    """
    Configuration of dataset UC executions
    """

    # the size of total features
    params = 10

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([0, 8]) #1 age
    input_bounds.append([0, 1]) #2 sex
    input_bounds.append([0, 5]) #3 race
    input_bounds.append([0, 34]) #4 state
    input_bounds.append([0, 3]) #5 region
    input_bounds.append([0, 4]) #6 method
    input_bounds.append([0, 1]) #7 juvenile
    input_bounds.append([0, 1]) #8 federal
    input_bounds.append([0, 1]) #9 volunteer
    input_bounds.append([0, 1]) #10 foreign_national

    # the name of each feature
    feature_name = ["age",
                    "sex",
                    "race",
                    "state",
                    "region",
                    "method",
                    "juvenile",
                    "federal",
                    "volunteer",
                    "foreign_national"]

    # the name of each class
    class_name = ["yes", "no"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


class compas:
    """
    Configuration of dataset UC executions
    """

    # the size of total features
    params = 12

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([0, 3]) #1 Agency_Text
    input_bounds.append([0, 1]) #2 Sex_Code_Text
    input_bounds.append([0, 8]) #3 Ethnic_Code_Text
    input_bounds.append([0, 1]) #4 ScaleSet
    input_bounds.append([0, 1]) #5 Language
    input_bounds.append([0, 6]) #6 LegalStatus
    input_bounds.append([0, 5]) #7 CustodyStatus
    input_bounds.append([0, 6]) #8 MaritalStatus
    input_bounds.append([0, 3]) #9 RecSupervisionLevel
    input_bounds.append([0, 2]) #10 DisplayText
    input_bounds.append([0, 60]) #11 RawScore
    input_bounds.append([0, 1]) #12 AssessmentType

    # the name of each feature
    feature_name = ["Agency_Text",
                    "Sex_Code_Text",
                    "Ethnic_Code_Text",
                    "ScaleSet",
                    "Language",
                    "LegalStatus",
                    "CustodyStatus",
                    "MaritalStatus",
                    "RecSupervisionLevel",
                    "DisplayText",
                    "RawScore",
                    "AssessmentType"]

    # the name of each class
    class_name = ["Low", "High"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]