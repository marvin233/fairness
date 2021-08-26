class census:
    # the size of total features
    params = 13

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([1, 9])
    input_bounds.append([0, 7])
    input_bounds.append([0, 39])
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


class bank:
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
    feature_name = ["age",
                    "job",
                    "marital",
                    "education",
                    "default",
                    "balance",
                    "housing",
                    "loan",
                    "contact",
                    "day",
                    "month",
                    "duration",
                    "campaign",
                    "pdays",
                    "previous",
                    "poutcome"]

    # the name of each class
    class_name = ["no", "yes"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


class compas:
    # the size of total features
    params = 14

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([0, 1])
    input_bounds.append([1, 9])
    input_bounds.append([0, 5])
    input_bounds.append([0, 20])
    input_bounds.append([0, 13])
    input_bounds.append([0, 11])
    input_bounds.append([0, 4])
    input_bounds.append([-6, 10])
    input_bounds.append([0, 90])
    input_bounds.append([0, 2])
    input_bounds.append([-1, 1])
    input_bounds.append([0, 2])
    input_bounds.append([0, 1])
    input_bounds.append([-1, 10])

    # the name of each feature
    feature_name = ["sex",
                    "age",
                    "race",
                    "juv_fel_count",
                    "juv_misd_count",
                    "juv_other_count",
                    "priors_count",
                    "days_b_screening_arrest",
                    "c_days_from_compas",
                    "c_charge_degree",
                    "is_recid",
                    "r_charge_degree",
                    "is_violent_recid",
                    "v_decile_score"]

    # the name of each class
    class_name = ["Low", "High"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]