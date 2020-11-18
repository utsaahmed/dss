import requests
import pandas
import scipy
import numpy
import sys


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


ef predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.
    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    decoded_content = response.content.decode('utf-8')

    df = csv.reader(decoded_content.splitlines(), delimiter=',')
    df=list(df)
    ar_train=list(map(float,df[0][1:]))
    pr_train=list(map(float,df[1][1:]))
    ar_train=numpy.array(ar_train)
    pr_train=numpy.array(pr_train)
    # YOUR IMPLEMENTATION HERE
    ...
    
    
    regre=LinearRegression()
    regre.fit(ar_train.reshape(-1,1),pr_train.reshape(-1,1))
    pr_pred=regre.predict(areas.reshape(-1,1))
    print(regre.intercept_)
    print(regre.coef_)
    
    return pr_pred

if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
