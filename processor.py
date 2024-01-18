import os
import sys
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from joblib import dump
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib


def get_base_path():
    if getattr(sys, 'frozen', False):
        return pathlib.Path(__file__).parent.parent.resolve()
    else:
        return os.path.dirname(os.path.abspath(sys.argv[0]))


path = get_base_path()
print(path)


def extract_train_data_and_drop():
    train_data_path = f'{path}/working/development.csv'
    print('start  extract_train_data_list')
    df = pd.read_csv(train_data_path)
    print('finish  extract_train_data_list')
    predicting_data_path = f'{path}\\working\\evaluation.csv'
    print('start   extract_predicting_data_list')
    predicting_df = pd.read_csv(predicting_data_path).iloc[:, 1:]
    print('finish  extract_predicting_data_list')

    columns_to_drop = ['pmax[0]', 'negpmax[0]', 'area[0]', 'tmax[0]', 'rms[0]',
                       'pmax[7]', 'negpmax[7]', 'area[7]', 'tmax[7]', 'rms[7]',
                       'pmax[12]', 'negpmax[12]', 'area[12]', 'tmax[12]', 'rms[12]',
                       'pmax[15]', 'negpmax[15]', 'area[15]', 'tmax[15]', 'rms[15]',
                       'pmax[16]', 'negpmax[16]', 'area[16]', 'tmax[16]', 'rms[16]',
                       'pmax[17]', 'negpmax[17]', 'area[17]', 'tmax[17]', 'rms[17]',]
    df = df.drop(columns=columns_to_drop)
    predicting_df = predicting_df.drop(columns=columns_to_drop)
    df_dependent = df.iloc[:, 0:2]
    df_independent = df.iloc[:, 2:]
    return df_dependent, df_independent, predicting_df


def extract_train_data_and_clean():
    train_data_path = f'{path}/working/development.csv'
    print('start  extract_train_data_list')
    df = pd.read_csv(train_data_path)
    print('finish  extract_train_data_list')
    predicting_data_path = f'{path}\\working\\evaluation.csv'
    print('start   extract_predicting_data_list')
    predicting_df = pd.read_csv(predicting_data_path).iloc[:, 1:]
    print('finish  extract_predicting_data_list')

    columns_to_clean = []
    for i in range(0, 17):
        columns_to_clean.append(f'pmax[{i}]')
        columns_to_clean.append(f'negpmax[{i}]')
        columns_to_clean.append(f'area[{i}]')
        columns_to_clean.append(f'tmax[{i}]')
        columns_to_clean.append(f'rms[{i}]')
        i += 1

    # Extract the selected columns
    selected_columns = df[columns_to_clean]
    print('start_ploting')
    sns.pairplot(selected_columns)
    plt.savefig(f'{path}/result/pairplot.png')

    print('start_action')
    # Apply EllipticEnvelope for outlier detection in selected columns
    envelope = EllipticEnvelope(contamination=0.3)
    outliers = envelope.fit_predict(selected_columns)
    print(outliers)

    # Create a mask to filter out rows with outliers
    outliers_mask = outliers != -1
    print(outliers_mask)

    # Filter the original DataFrame based on the mask
    df_cleaned = df[outliers_mask]
    predicting_df_cleaned = predicting_df.iloc[outliers_mask]
    print(df_cleaned)

    df_dependent = df_cleaned.iloc[:, 0:2]
    df_independent = df_cleaned.iloc[:, 2:]
    return df_dependent, df_independent, predicting_df_cleaned


def linear_model_processor(custom_train_data_list):
    print('method: linear_model_processor')
    X = custom_train_data_list[1]
    y = custom_train_data_list[0]
    X_predict = custom_train_data_list[2]
    print(X_predict)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=57)

    linear_model = MultiOutputRegressor(LinearRegression())
    linear_model.fit(X_train, y_train)
    linear_model_predictions = linear_model.predict(X_test)
    linear_model_predictions_mse = mean_squared_error(y_test, linear_model_predictions)
    print(f'linear_model Mean Squared Error: {linear_model_predictions_mse}')


    predicting_data_predictions = linear_model.predict(X_predict)
    print(f'linear_model_predictions: {predicting_data_predictions}')
    result_data_frame = pd.DataFrame(predicting_data_predictions, columns=['x', 'y'])
    result_data_frame = result_data_frame.round(1)
    result_data_frame['id'] = result_data_frame.index
    print(result_data_frame)
    csv_export = result_data_frame[['id', 'x', 'y']].copy()
    csv_export['Predicted'] = csv_export[['x', 'y']].apply(lambda x: '|'.join(map(str, x)), axis=1)
    csv_export.drop(['x', 'y'], axis=1, inplace=True)
    csv_export.to_csv(f'{path}\\result\\linear_model_predicted_results.csv', index=False)

    dump(linear_model, f'{path}\\result\\linear_model_trained_model.joblib')


def model_rf_processor(custom_train_data_list):
    X = custom_train_data_list[0]
    y = custom_train_data_list[1]
    X_predict = custom_train_data_list[2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=57)
    model_rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, verbose=2, n_jobs=-1))
    model_rf.fit(X_train, y_train)
    predictions_rf = model_rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, predictions_rf)
    print(f'model_rf Mean Squared Error: {mse_rf}')

    predicting_data_predictions = model_rf.predict(X_predict)
    print(f'model_rf: {predicting_data_predictions}')

    result_data_frame = pd.DataFrame(predicting_data_predictions, columns=['x', 'y'])
    result_data_frame = result_data_frame.round(1)
    result_data_frame['id'] = result_data_frame.index
    csv_export = result_data_frame[['id', 'x', 'y']].copy()
    csv_export['Predicted'] = csv_export[['x', 'y']].apply(lambda x: '|'.join(map(str, x)), axis=1)
    csv_export.drop(['x', 'y'], axis=1, inplace=True)
    csv_export.to_csv(f'{path}\\result\\model_rf_predicted_results.csv', index=False)

    dump(model_rf, f'{path}\\result\\model_rf_trained_model.joblib')


if __name__ == '__main__':
    while True:
        print('please choose a number:')
        print('1. linear_model_processor(drop column manually)')
        print('2. linear_model_processor(clean data)')
        print('3. model_rf_processor(drop column manually)')
        print('4. model_rf_processor(clean data)')
        number_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        try:
            number = int(input())
            if number in number_list:
                if number == 1:
                    try:
                        linear_model_processor(extract_train_data_and_drop())
                    except Exception as e:
                        print(f'exception: {str(e)}')
                    break
                elif number == 2:
                    try:
                        linear_model_processor(extract_train_data_and_clean())
                    except Exception as e:
                        print(f'exception: {str(e)}')
                    break
                elif number == 3:
                    try:
                        model_rf_processor(extract_train_data_and_drop())
                    except Exception as e:
                        print(f'exception: {str(e)}')
                    break
                elif number == 4:
                    try:
                        model_rf_processor(extract_train_data_and_clean())
                    except Exception as e:
                        print(f'exception: {str(e)}')
                    break
            else:
                print('not correct. please choose again:')
        except:
            print('not correct. please choose again:')
