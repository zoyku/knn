import pandas as pd

def knn(existing_data: pd.DataFrame, test_data: pd.DataFrame, k: int, distance_method: str, re_training: bool, distance_threshold: float, weighted_voting: bool):
    
    string_data = pd.read_csv(existing_data)
    string_data = pd.read_csv(test_data)

    return 0