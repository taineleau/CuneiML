import numpy as np
from sklearn.metrics import classification_report
import json
import random

with open('/graft3/code/tracy/data/final_may24_ver2/test_data.json', 'r') as file:
    test_data = json.load(file)
with open('/graft3/code/tracy/data/final_may24_ver2/test_data_2.json', 'r') as file:
    test_data2 = json.load(file)
with open('/graft3/code/tracy/data/final_may24_ver2/test_data_3.json', 'r') as file:
    test_data3 = json.load(file)


test_labels = [entry['time'] for entry in test_data.values()]
test_labels2 = [entry['time'] for entry in test_data2.values()]
test_labels3 = [entry['time'] for entry in test_data3.values()]

# List of historical periods to be used as class labels
label_list = [
    'Ur III (ca. 2100-2000 BC)', 'Old Babylonian (ca. 1900-1600 BC)',
    'Old Akkadian (ca. 2340-2200 BC)', 'ED IIIb (ca. 2500-2340 BC)',
    'Old Assyrian (ca. 1950-1850 BC)', 'Early Old Babylonian (ca. 2000-1900 BC)',
    'Neo-Assyrian (ca. 911-612 BC)', 'Middle Babylonian (ca. 1400-1100 BC)',
    'ED IIIa (ca. 2600-2500 BC)', 'Middle Assyrian (ca. 1400-1000 BC)',
    'Ebla (ca. 2350-2250 BC)', 'Lagash II (ca. 2200-2100 BC)',
    'Neo-Babylonian (ca. 626-539 BC)', 'ED I-II (ca. 2900-2700 BC)'
]

# Generate random predictions
random_predictions = [random.choice(label_list) for _ in test_labels]
random_predictions2 = [random.choice(label_list) for _ in test_labels2]
random_predictions3 = [random.choice(label_list) for _ in test_labels3]


# Compute classification metrics
report = classification_report(
    test_labels, random_predictions, output_dict=True)
# max_f1_random = max(class_label['f1-score'] for class_label in report.values() if isinstance(class_label, dict))
macro_f1_random = report['macro avg']['f1-score']
# Micro F1-score is equivalent to accuracy in this context
micro_f1_random = report['accuracy']

report2 = classification_report(
    test_labels2, random_predictions2, output_dict=True)
macro_f1_random2 = report2['macro avg']['f1-score']
# Micro F1-score is equivalent to accuracy in this context
micro_f1_random2 = report2['accuracy']

# Compute classification metrics for test_data3
report3 = classification_report(
    test_labels3, random_predictions3, output_dict=True)
macro_f1_random3 = report3['macro avg']['f1-score']
micro_f1_random3 = report3['accuracy']

# print(f"Random - Maximum F1-score across classes: {max_f1_random:.4f}")
print(f"Random - Macro F1-score: {macro_f1_random:.4f}")
print(f"Random - Micro F1-score: {micro_f1_random:.4f}")
print(f"Test Data 2 - Random - Macro F1-score: {macro_f1_random2:.4f}")
print(f"Test Data 2 - Random - Micro F1-score: {micro_f1_random2:.4f}")
print(f"Test Data 3 - Random - Macro F1-score: {macro_f1_random3:.4f}")
print(f"Test Data 3 - Random - Micro F1-score: {micro_f1_random3:.4f}")
