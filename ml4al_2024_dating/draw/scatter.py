import matplotlib.pyplot as plt
import seaborn as sns

# Setting a seaborn style
sns.set_style("whitegrid")

# Data
num_lines = [1, 2, 4, 8, 16, 32]
translit_f1_test1 = [15.77, 16.14, 15.53, 14.06, 12.92, 12.96]
translit_f1_test2 = [26.52, 25.51, 25.61, 25.14, 18.37, 18.22]
translit_f1_test3 = [74.89, 84.42, 78.66, 81.53, 73.17, 80.24]
sign_f1_test1 = [15.00, 14.58, 15.45, 14.40, 14.32, 16.59]
sign_f1_test2 = [19.14, 18.26, 17.57, 22.58, 23.11, 24.25]
sign_f1_test3 = [67.56, 75.44, 78.13, 72.82, 74.57, 74.15]

# Plotting
plt.figure(figsize=(12, 8))

# Transliteration
plt.plot(num_lines, translit_f1_test1, 'o-', color='navy',
         label='Test 1 + Transliteration', markersize=8, linewidth=2)
plt.plot(num_lines, translit_f1_test2, 'o-', color='forestgreen',
         label='Test 2 + Transliteration', markersize=8, linewidth=2)
plt.plot(num_lines, translit_f1_test3, 'o-', color='firebrick',
         label='Test 3 + Transliteration', markersize=8, linewidth=2)

# Sign token
plt.plot(num_lines, sign_f1_test1, '*--', color='dodgerblue',
         label='Test 1 + Sign Token', markersize=8, linewidth=2)
plt.plot(num_lines, sign_f1_test2, '*--', color='limegreen',
         label='Test 2 + Sign Token', markersize=8, linewidth=2)
plt.plot(num_lines, sign_f1_test3, '*--', color='crimson',
         label='Test 3 + Sign Token', markersize=8, linewidth=2)

plt.xlabel('Number of Lines', fontsize=14)
plt.ylabel('Macro F1 Score (%)', fontsize=14)
plt.title('Macro F1 Scores for Different Tests and Methods',
          fontsize=16, fontweight='bold')
plt.legend(loc='best', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# Save the figure to PDF
plt.savefig('Macro_F1_Scores_Professional.pdf')

plt.show()
