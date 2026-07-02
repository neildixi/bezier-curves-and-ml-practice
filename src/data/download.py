from ucimlrepo import fetch_ucirepo 
import pandas as pd
import os
# fetch dataset 
pen_based_recognition_of_handwritten_digits = fetch_ucirepo(id=81) 
  
# data (as pandas dataframes) 
X = pen_based_recognition_of_handwritten_digits.data.features 
y = pen_based_recognition_of_handwritten_digits.data.targets 
  
# metadata 
#print(pen_based_recognition_of_handwritten_digits.metadata) 
  
# variable information 
#print(pen_based_recognition_of_handwritten_digits.variables) 

def download_pendigits(output_dir):
    assert X.shape[1] ==16, '''X must have 16 features'''

    X.to_csv(os.path.join(output_dir, 'pendigits_features.csv'), index=False)

    y.to_csv(os.path.join(output_dir, 'pendigits_labels.csv'), index=False)

    return X, y
    
    '''with open(os.path.join(output_dir, 'pendigits_features.csv'), w) as file: 
        writer = csv.writer(file)
        writer.writerow(X.columns)
        for row in X.values:
            writer.writerow(row)
    
    with open(os.path.join(output_dir, 'pendigits_labels.csv'), w) as file_2:
        writer_2 = csv.writer(file_2)
        writer_2.writerow(y.columns)
        for row in y.values:
            writer.writerow(row) '''
output_directory = 'C:/Users/neild/OneDrive/Desktop/Published Research Papers/Bezier Curves ML/bezier-curves-and-ml-practice/data/raw'
download_pendigits(output_directory)


    



