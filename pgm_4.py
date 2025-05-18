import pandas as pd

# Make sure the file path is complete and correctly formatted
data = pd.read_csv(r"file path")

print(data)

import pandas as pd

def find_s_algorithm(data):
    attributes = data.iloc[:, :-1].values 
    target = data.iloc[:, -1].values      
    hypothesis = None
    for i in range(len(target)):
        if target[i].lower() == "yes":  
            hypothesis = attributes[i].copy()
            break
    if hypothesis is None:
        return "No positive examples found."
    for i in range(len(target)):
        if target[i].lower() == "yes":
            for j in range(len(hypothesis)):
                if hypothesis[j] != attributes[i][j]:
                    hypothesis[j] = '?' 

    return hypothesis
final_hypothesis = find_s_algorithm(data)
print("Most Specific Hypothesis:", final_hypothesis)

