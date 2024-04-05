import pandas as pd
import numpy as np


def euclidean_distance(x, y):
    sum_row = 0
    for i in range(1,31):
        sum_row = sum_row + (x[i] - y[i])**2
    return np.sqrt(sum_row)

def manhattan_distance(x, y):
    sum_row = 0
    for i in range(1,31):
        sum_row = sum_row + np.abs(x[i] - y[i])
    return np.sum(sum_row)

def chebyshev_distance(x, y):
    dist = 0
    for i in range(1,31):
        dist = np.max(dist, x[i] - y[i])
    return dist


def distance_function(x, y, method):
    if method == 'euclidean':
        return euclidean_distance(x, y)
    elif method == 'manhattan':
        return manhattan_distance(x, y)
    elif method == 'chebyshev':
        return chebyshev_distance(x, y)
    

def knn(existing_data: pd.DataFrame, test_data: pd.DataFrame, k: int,
        distance_method: str, re_training: bool, distance_threshold: float, weighted_voting: bool):
    
    predictions = []

    # string_data_train = pd.read_csv(existing_data)
    # string_data_test = pd.read_csv(test_data)

    for index, test_row in test_data.iterrows():
        distance_with_labels = []
        for index, train_row in existing_data.iterrows():
            distance = distance_function(train_row, test_row, distance_method)
            distance_with_labels.append((distance, train_row[0]))

        if re_training:
            new_row = test_row.copy()
            # string_data_train = string_data_train.append(new_row, ignore_index=True)
            # string_data_train = pd.concat([string_data_train, new_row])
            existing_data.loc[len(existing_data)] = new_row

        # print(string_data_train.tail(n=1))

        distance_with_labels = sorted(distance_with_labels, key=lambda x: x[0])

        if distance_threshold is not None:
            for x in distance_with_labels:
                if x[0] > distance_threshold:
                    distance_with_labels.remove(x)
        
        neighbors = distance_with_labels[:k]

        count_label0 = 0
        count_label1 = 0

        if weighted_voting:
            for x in neighbors:
                x[0] = 1 / (x[0])**2
    
            for d in neighbors:
                if d[1] == 0:
                    count_label0 += d[0]
                elif d[1] == 1:
                    count_label1 += d[0]
        else:
            for d in neighbors:
                if d[1] == 0:
                    count_label0 += 1
                elif d[1] == 1:
                    count_label1 += 1

        if count_label0 > count_label1:
            predictions.append(0)
            # print("0")
        else:
            predictions.append(1)
            # print("1")

    series_prediction = pd.Series(predictions) 
    print(series_prediction)

    return series_prediction


def fill_missing_features(existing_data: pd.DataFrame, test_data: pd.DataFrame,
                            k: int, distance_method: str, distance_threshold: float, weighted_voting: bool):
    
    for index, test_row in test_data.iterrows():
        distance_with_labels = []
        missing_features = test_row[test_row.isna()].index
        modified_test_row = test_row.drop(columns=[missing_features])
        for index, train_row in existing_data.iterrows():
            modified_train_row = train_row.drop(columns=[missing_features])
            distance = distance_function(modified_train_row, modified_test_row, distance_method)
            distance_with_labels.append((distance, train_row[0]))

        distance_with_labels = sorted(distance_with_labels, key=lambda x: x[0])
 
       # TODO: Implement the rest of the function 

    return 0


def main():

    train_data = pd.read_csv('train_normalized_v2.csv')
    test_data = pd.read_csv('test_normalized_v2.csv')
    test_with_missing = pd.read_csv('test_with_missing_normalized_v2.csv')

    k = 5  
    distance_method = "euclidean"
    re_training = True 
    distance_threshold = None 
    weighted_voting = False 

    knn(train_data, test_data, k, distance_method, re_training, distance_threshold, weighted_voting)
    # predictions = knn(train_data, test_data, k, distance_method, re_training, distance_threshold, weighted_voting)
    # print("kNN Predictions:", predictions)

    # fill_missing_features(train_data, test_with_missing, k, distance_method, distance_threshold, weighted_voting)
    # filled_data = fill_missing_features(train_data, test_with_missing, k, distance_method, distance_threshold, weighted_voting)
    # print("Filled Test Data:", filled_data)


if __name__ == "__main__":
    main()