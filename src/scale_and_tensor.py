from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch

def scale_and_tensor(X_train, y_train, X_test, y_test ): 
    
    #Scale features and convert them to tensors
    scaler_x = StandardScaler()
    scaler_x.fit(X_train)
    X_train_tensor = torch.tensor(scaler_x.transform(X_train))

    scaler_x = StandardScaler()
    scaler_x.fit(X_test)
    X_test_tensor = torch.tensor(scaler_x.transform(X_test))

    scaler_y_train = MinMaxScaler(feature_range=(-1, 1)).fit(y_train.to_numpy().reshape(-1, 1))
    y_train_tensor = torch.tensor(scaler_y_train.transform(y_train.to_numpy().reshape(-1, 1)))

    scaler_y_test = MinMaxScaler(feature_range=(-1, 1)).fit(y_test.to_numpy().reshape(-1, 1))
    y_test_tensor = torch.tensor(scaler_y_test.transform(y_test.to_numpy().reshape(-1, 1)))

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler_y_train, scaler_y_test