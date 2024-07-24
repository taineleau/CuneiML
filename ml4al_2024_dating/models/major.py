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

prediction = 'Ur III (ca. 2100-2000 BC)'
# Generate fixed predictions for all entries in each dataset
fixed_predictions = [prediction] * len(test_labels)
fixed_predictions2 = [prediction] * len(test_labels2)
fixed_predictions3 = [prediction] * len(test_labels3)

# Compute classification metrics for each dataset
report = classification_report(
    test_labels, fixed_predictions, output_dict=True)
macro_f1_fixed = report['macro avg']['f1-score']
# Micro F1-score is equivalent to accuracy in this context
micro_f1_fixed = report['accuracy']

report2 = classification_report(
    test_labels2, fixed_predictions2, output_dict=True)
macro_f1_fixed2 = report2['macro avg']['f1-score']
micro_f1_fixed2 = report2['accuracy']

report3 = classification_report(
    test_labels3, fixed_predictions3, output_dict=True)
macro_f1_fixed3 = report3['macro avg']['f1-score']
micro_f1_fixed3 = report3['accuracy']

# Print results
print(f"Test Data 1 - Fixed Prediction - Macro F1-score: {macro_f1_fixed:.4f}")
print(f"Test Data 1 - Fixed Prediction - Micro F1-score: {micro_f1_fixed:.4f}")

print(
    f"Test Data 2 - Fixed Prediction - Macro F1-score: {macro_f1_fixed2:.4f}")
print(
    f"Test Data 2 - Fixed Prediction - Micro F1-score: {micro_f1_fixed2:.4f}")

print(
    f"Test Data 3 - Fixed Prediction - Macro F1-score: {macro_f1_fixed3:.4f}")
print(
    f"Test Data 3 - Fixed Prediction - Micro F1-score: {micro_f1_fixed3:.4f}")
