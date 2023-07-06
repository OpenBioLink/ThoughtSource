import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt

def predictions(data,idx_cot,dataset,split,plot_title):
    scores_with_indicator= parse_data_test(data,idx_cot=1,dataset='med_qa',split='test')
    prediction_results(scores_with_indicator,plot_title)

def parse_data_test(new_data,idx_cot,dataset,split):

    scores_with_indicator = []
    for item in new_data[dataset][split]:
        flag = False
        try:
            data = yaml.load(item['generated_cot'][idx_cot]['cot'], Loader=yaml.FullLoader)
            if any(isinstance(v, str) for v in data.values()):
                new_dict = {k: v for k, v in data.items() if k.startswith('obj')}
                #scores_per_item.append(new_dict)
                flag = True

        except:
            continue

        if flag == True:
            data = new_dict
            
            try:
                # Find the lowest value
                lowest_value = min(data.values())

                # Calculate the average value
                average_value = sum(data.values()) / len(data)

                # Add lowest and average values to the dictionary
                data['lowest'] = lowest_value
                data['average'] = average_value
            except:
                data['lowest'] = 'None'
                data['average'] = 'None'
   

            #add to scores_w_indicator without string data
            int_data = copy.deepcopy(data)

            scores_with_indicator.append((int_data,item['generated_cot'][0]['answers'][0]['correct_answer']))

    return scores_with_indicator


# Modify the function to handle 'None' as string
def handle_string(df, column):
    for index, value in df[column].items():
        if isinstance(value, str):
            if value.lower() == 'none':
                df.loc[index, column] = None  # or np.nan
            else:
                continue
import pandas as pd

def prediction_results(scores_with_indicator,plot_title):
    data = scores_with_indicator
    data = [(t[0], False if t[1] is None else t[1]) for t in data]

    #all objs
    df = pd.DataFrame([t[0] for t in data])
    df['Indicator'] = [t[1] for t in data]

    #df_for_regression does not include variables created later 
    df_for_reg = copy.deepcopy(df)
    handle_string(df_for_reg,'average')
    handle_string(df,'average')

    df = df[df['average'].notna()]
    df = df.sort_values(by='average')

    df['Cumulative Count'] = df['Indicator'].cumsum()

    # Create a new column 'Cumulative Count False' that contains the cumulative count of 'False' values
    df['Cumulative Count False'] = (~df['Indicator']).cumsum()

    # Plot graph with 'Average' as x-axis and 'Cumulative Count' and 'Cumulative Count False' as y-axis
    plt.figure(figsize=(10, 6))
    plt.plot(df['average'], df['Cumulative Count'], label='Correct')
    plt.plot(df['average'], df['Cumulative Count False'], label='Incorrect', color='red')
    plt.xlabel('Average Objective Score ') #Distribution of (in)correct answers when average objective score increases
    plt.ylabel('Cumulative Sum of Questions Evaluated')
    plt.title(plot_title)
    plt.grid(True)
    plt.legend()
    plt.show()

    minimum_graph(scores_with_indicator,plot_title)

    df['Difference'] = df['Cumulative Count False'] - df['Cumulative Count']

    # Find the maximum value in 'Difference'
    max_diff = df['Difference'].max()


    #out of sample accuracy
    df_for_reg = df_for_reg[df_for_reg['average'].notna()]

    for column in df_for_reg.columns:
        if column.startswith('obj_'):
            is_float = df_for_reg[column].apply(lambda x: not isinstance(x, str))
            df_for_reg = df_for_reg[is_float]

    X = df_for_reg.drop('Indicator', axis=1)  # Features
    y = df_for_reg['Indicator'].values  # Target variable

    
    loo = LeaveOneOut()
    classifier = LogisticRegression(max_iter=10000)

    accuracies = []  # List to store all accuracies
    y_true = []  # List to store true labels
    y_preds = []  # List to store predicted labels

    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        # Store true and predicted labels
        y_true.extend(y_test)
        y_preds.extend(y_pred)
    
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_preds).ravel()
    print("Confusion matrix:")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}\n")


    # Print the average accuracy
    print("Average Accuracy by LeaveOneOut strategy:", np.mean(accuracies))

    # training accuracy:
    classifier = LogisticRegression(max_iter=10000)
    classifier.fit(X, y)

    y_pred = classifier.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print("Training Accuracy:", accuracy)

    # Calculate the correlation matrix
    correlation_matrix = df.corr()

    # Get the correlation values with the 'Indicator' variable
    indicator_correlation = correlation_matrix['Indicator']
    print("\nCorrelations")
    print(indicator_correlation)
    print("\n")

def minimum_graph(scores_with_indicator,plot_title):
    data = scores_with_indicator
    data = [(t[0], False if t[1] is None else t[1]) for t in data]

    #all objs
    df = pd.DataFrame([t[0] for t in data])
    df['Indicator'] = [t[1] for t in data]

    #df_for_regression does not include variables created later 
    df_for_reg = copy.deepcopy(df)
    handle_string(df_for_reg,'lowest')
    handle_string(df,'lowest')

    df = df[df['lowest'].notna()]
    df = df.sort_values(by='lowest')

    df['Cumulative Count'] = df['Indicator'].cumsum()

    # Create a new column 'Cumulative Count False' that contains the cumulative count of 'False' values
    df['Cumulative Count False'] = (~df['Indicator']).cumsum()

    # Plot graph with 'Average' as x-axis and 'Cumulative Count' and 'Cumulative Count False' as y-axis
    plt.figure(figsize=(10, 6))
    plt.plot(df['lowest'], df['Cumulative Count'], label='Correct')
    plt.plot(df['lowest'], df['Cumulative Count False'], label='Incorrect', color='red')
    plt.xlabel('Minimum Objective Score ') #Distribution of (in)correct answers when average objective score increases
    plt.ylabel('Cumulative Sum of Questions Evaluated')
    plt.title(plot_title)
    plt.grid(True)
    plt.legend()
    plt.show()