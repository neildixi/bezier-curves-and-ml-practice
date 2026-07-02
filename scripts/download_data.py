if __name__ == "__main__":
    from pathlib import Path
    import sys
    ROOT = Path(__file__).resolve().parents[1] 
    sys.path.insert(0, str(ROOT))
    from src.data.download import download_pendigits
    output_dir = ROOT / 'data' / 'raw'
    X, y = download_pendigits(output_dir)
    print(X.shape, y.shape)
    print(y.iloc[0])
    #Checks: 
    print('Here are some quick sanity checks:')
    print(f'Does the data have 16 features: {X.shape[1]==16}')
    print(f'Does the data have the same length as the labels: {X.shape[0]==y.shape[0]}')
    labels = y["Class"]
    unique_labels = set(labels.unique())
    print(f'Does the data have the right set of labels: {unique_labels=={0,1,2,3,4,5,6,7,8,9}}')
    values = X.to_numpy()
    min_val = values.min()
    max_val = values.max()
    print(f'Are the coordinates in the range of 0-100: {min_val >= 0 and max_val <= 100}')
    first_row = X.iloc[0]
    first_row_coordinates = []
    for i in range(len(first_row)//2):
        first_row_coordinates.append((first_row.iloc[2*i], first_row.iloc[2*i+1]))
    print(f'Here is the first feature (set of 8 tuples): {first_row_coordinates}')

    