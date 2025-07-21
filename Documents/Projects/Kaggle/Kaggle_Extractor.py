def download_and_load_all_csvs(dataset_name, base_path='kaggle_data'):
    import os
    import zipfile
    import pandas as pd
    from kaggle.api.kaggle_api_extended import KaggleApi

    # Create dataset-specific subfolder
    dataset_id = dataset_name.split('/')[-1]
    download_path = os.path.join(base_path, dataset_id)
    os.makedirs(download_path, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    zip_target = os.path.join(download_path, f"{dataset_id}.zip")
    if not os.path.exists(zip_target):
        print(f"Downloading dataset: {dataset_name}")
        api.dataset_download_files(dataset_name, path=download_path, unzip=False)

    # Look only in this dataset's folder
    zip_files = [f for f in os.listdir(download_path) if f.endswith('.zip')]
    if not zip_files:
        raise FileNotFoundError("No ZIP file found.")
    zip_path = os.path.join(download_path, zip_files[0])

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(download_path)

    # Now list only extracted CSVs for this dataset
    all_files = os.listdir(download_path)
    csv_files = [f for f in all_files if f.endswith('.csv')]

    if not csv_files:
        raise FileNotFoundError("No CSV files found in this dataset.")

    dataframes = {}
    for file in csv_files:
        path = os.path.join(download_path, file)
        print(f"Loading {file}")
        dataframes[file] = pd.read_csv(path)

    return dataframes

# Example usage:
#dataset = 'swatikhedekar/exploratory-data-analysis-on-netflix-data'
#df = download_and_load_all_csvs(dataset)
#print(df.head())
