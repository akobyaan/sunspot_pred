from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def accuracies(y_test, y_pred):
    mean_squared = mean_squared_error(y_test, y_pred)
    mean_absolute = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean_squared_error is: {mean_squared}")
    print(f"Mean_absolute_error is: {mean_absolute}")
    print(f"r2 is: {r2}")
