from src import data_importing, data_preparing, model_training, model_accuracy


def main():
    file_path = 'data/sunspot_data.csv'

    # Importing Data
    df = data_importing.import_data(file_path)

    # Get Data information and Split the columns
    data_preparing.data_information(df)
    X, y = data_preparing.ind_dep_columns(df)

    # Train the model
    y_test, y_pred = model_training.train_model(X, y)

    # Get model accuracies
    model_accuracy.accuracies(y_test, y_pred)


if __name__ == "__main__":
    main()