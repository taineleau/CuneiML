from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import numpy as np
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB, CategoricalNB
from scipy.sparse import hstack
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import json
import pandas as pd

with open('/graft3/code/tracy/data/final_may24_ver2/train_data.json', 'r') as file:
    train_data = json.load(file)
with open('/graft3/code/tracy/data/final_may24_ver2/valid_data.json', 'r') as file:
    valid_data = json.load(file)

with open('/graft3/code/tracy/data/final_may24_ver2/test_data.json', 'r') as file:
    test_data = json.load(file)
with open('/graft3/code/tracy/data/final_may24_ver2/test_data_2.json', 'r') as file:
    test_data2 = json.load(file)
with open('/graft3/code/tracy/data/final_may24_ver2/test_data_3.json', 'r') as file:
    test_data3 = json.load(file)

file_path = '/graft3/code/tracy/data/expanded_catalogue.csv'
df = pd.read_csv(file_path)


def get_museums(data):
    museums = []
    for i in data:
        try:
            row_with_id = df[df['id'] == int(i)]
            collections_data_str = row_with_id['collections'].iloc[0]
            collections_data = ast.literal_eval(collections_data_str)
            if collections_data:
                museums.append(str(collections_data[0]['collection_id']))
            else:
                museums.append('Unknown')
        except KeyError:
            print(i)
    return museums


def get_proveniences(data):
    proveniences = []
    for i in data:
        try:
            prov_id = df[df['id'] == int(i)]['provenience.id'].iloc[0]
            if pd.isna(prov_id):
                # Append 'Unknown' if the value is NaN
                proveniences.append('Unknown')
            else:
                proveniences.append(str(int(prov_id)))
        except KeyError:
            print(i)
    return proveniences


def get_genres(data):
    genres = []
    for i in data:
        data_str = df[df['id'] == int(i)]['genres'].iloc[0]
        genre = ast.literal_eval(data_str)
        raw_genre = str(genre)
        # weird string in comments, remove it for simplity
        raw_genre = re.sub(r'\'comments\': ["][^"]*["],', '', raw_genre)
        raw_genre = re.sub(r'\'comments\': [\'][^\']*[\'],', '', raw_genre)
        raw_genre = re.sub(
            r"('comments':\s*'.*?'\s*,\s*)('genre':)", "'genre':", raw_genre)
        raw_genre = raw_genre.replace("True", 'true')
        genre = raw_genre.replace("'", '"').replace("True", 'true')
        try:
            ret = json.loads(genre)
        except:
            print(raw_genre)
            print(genre)
            raise KeyError
        if len(ret) > 0:
            g = ret[0]['genre']['genre']
        else:
            g = "Unknown"
        genres.append(g)
    return genres


def get_measurements(data):
    measurements = []
    for i in data:
        try:
            # 'thickness', 'height', 'width'
            thickness = df[df['id'] == int(i)]['thickness'].iloc[0]
            height = df[df['id'] == int(i)]['height'].iloc[0]
            width = df[df['id'] == int(i)]['width'].iloc[0]
            # range[0, 7623000]
            measurement = thickness * height * width
            if measurement == 0:
                measurement = 7623000/2
            measurements.append(measurements)
        except KeyError:
            print("err!")
            break
    print(max_m)
    return measurements


def prepare_features(data, feature_configs, fit=False):
    """
    Prepare feature matrix by processing and vectorizing multiple types of data.

    Args:
    data (dict): Dataset containing input data.
    feature_configs (list of dicts): Each dict contains:
        - 'feature_name' (str): Name of the feature in the data dict.
        - 'extractor_function' (callable): Function to extract the feature from data.
        - 'vectorizer' (Transformer like TfidfVectorizer or OneHotEncoder): Preprocessing instance.
    fit (bool): If True, fit the vectorizer, otherwise just transform.
    Returns:
    scipy.sparse matrix: Combined feature matrix.
    """
    features = []
    for config in feature_configs:
        feature_data = config['extractor_function'](data)
        if config['type'] == 'text':
            if fit:
                processed_feature = config['vectorizer'].fit_transform(
                    feature_data)
            else:
                processed_feature = config['vectorizer'].transform(
                    feature_data)
        elif config['type'] == 'numeric':
            scaler = StandardScaler()
            feature_data_array = np.array(feature_data)
            if fit:
                processed_feature = scaler.fit_transform(
                    feature_data_array.reshape(-1, 1))
            else:
                processed_feature = scaler.transform(
                    feature_data_array.reshape(-1, 1))
        elif config['type'] == 'cat':
            if fit:
                counter = Counter(feature_data)
                counter['Unknown'] = 0
                config['mapping'] = {x: idx for idx,
                                     x in enumerate(sorted(counter.keys()))}
            processed_feature = [config['mapping'].get(
                f, config['mapping']['Unknown']) for f in feature_data]
            processed_feature = np.atleast_2d(processed_feature).T
        else:
            if fit:
                counter = Counter(feature_data)
                counter['Unknown'] = 0
                config['mapping'] = {x: idx for idx,
                                     x in enumerate(sorted(counter.keys()))}
            processed_feature = [config['mapping'].get(
                f, config['mapping']['Unknown']) for f in feature_data]
            processed_feature = np.atleast_2d(processed_feature).T
            features.append(processed_feature)

        features.append(processed_feature)

    if len(features) > 1:
        return np.hstack(features)
    else:
        return features[0]


feature_configs_4 = [
    {
        'feature_name': 'museum',
        'extractor_function': get_museums,
        'mapping': {},
        'type': 'cat'
    },
]

feature_configs_5 = [
    # {
    #     'feature_name': 'provenience',
    #     'extractor_function': get_proveniences,
    #     'mapping': {},
    #     'type': 'cat'
    # },
    {
        'feature_name': 'genre',
        'extractor_function': get_genres,
        'mapping': {},
        'type': 'cat'
    },
]

use_features = feature_configs_4

X_train = prepare_features(train_data, use_features, fit=True)
X_valid = prepare_features(valid_data, use_features, fit=False)
X_test = prepare_features(test_data, use_features, fit=False)
X_test2 = prepare_features(test_data2, use_features, fit=False)
X_test3 = prepare_features(test_data3, use_features, fit=False)

train_labels = [entry['time'] for entry in train_data.values()]
valid_labels = [entry['time'] for entry in valid_data.values()]
test_labels = [entry['time'] for entry in test_data.values()]
test_labels2 = [entry['time'] for entry in test_data2.values()]
test_labels3 = [entry['time'] for entry in test_data3.values()]


clf = CategoricalNB()
clf.fit(X_train, train_labels)

predictions_train = clf.predict(X_train)
predictions_valid = clf.predict(X_valid)
predictions_test = clf.predict(X_test)
predictions_test2 = clf.predict(X_test2)
predictions_test3 = clf.predict(X_test3)

print("Train - Macro F1 Score:", classification_report(train_labels,
      predictions_train, output_dict=True)['macro avg']['f1-score'])
print("Train - Micro F1 Score:", classification_report(train_labels,
      predictions_train, output_dict=True)['accuracy'])

print("Validation - Macro F1 Score:", classification_report(valid_labels,
      predictions_valid, output_dict=True)['macro avg']['f1-score'])
print("Validation - Micro F1 Score:", classification_report(valid_labels,
      predictions_valid, output_dict=True)['accuracy'])

print("Test - Macro F1 Score:", classification_report(test_labels,
      predictions_test, output_dict=True)['macro avg']['f1-score'])
print("Test - Micro F1 Score:", classification_report(test_labels,
      predictions_test, output_dict=True)['accuracy'])

print("Test2 - Macro F1 Score:", classification_report(test_labels2,
      predictions_test2, output_dict=True)['macro avg']['f1-score'])
print("Test2 - Micro F1 Score:", classification_report(test_labels2,
      predictions_test2, output_dict=True)['accuracy'])

print("Test3 - Macro F1 Score:", classification_report(test_labels3,
      predictions_test3, output_dict=True)['macro avg']['f1-score'])
print("Test3 - Micro F1 Score:", classification_report(test_labels3,
      predictions_test3, output_dict=True)['accuracy'])
