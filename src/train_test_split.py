def train_test_split(df, id_train, id_test, x_min, x_max, y_min, y_max, input_cols, output_col):
    
    
    if isinstance(id_train, int):
        id_train = [id_train]
    train_df = df[df['exp_id'].isin(id_train)]
    
    if isinstance(id_test, int):
        id_test = [id_test]
    test_df = df[df['exp_id'].isin(id_test)]


    train_df_zone = train_df[(train_df['x'] >= x_min) & (train_df['x'] <= x_max) & (train_df['y'] >= y_min) & (train_df['y'] <= y_max)]

    test_df_zone = test_df[(test_df['x'] >= x_min) & (test_df['x'] <= x_max) & (test_df['y'] >= y_min) & (test_df['y'] <= y_max)]

    X_train = train_df_zone[input_cols]
    y_train = train_df_zone[output_col]
    
    X_test = test_df_zone[input_cols]
    y_test = test_df_zone[output_col]
    
    return X_train, y_train, X_test, y_test


