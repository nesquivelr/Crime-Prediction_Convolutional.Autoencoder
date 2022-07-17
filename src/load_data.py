"""
This module load the data, then transform it to a list of matrices
"""
import multiprocessing as mp
from functools import partial
import pickle
import pandas as pd
import numpy as np
import numpy.typing as npt
from tqdm import tqdm


def load_dataframe(file_path: str, crime: str) -> (pd.DataFrame, list[str]):
    """
    Creates a dataframe, remove unwanted columns and returns the dataframe
    """
    print("Loading data")
    data = pd.read_csv(file_path)
    data.dropna(inplace=True)
    crime_column = "Primary Type"
    data = data[data[crime_column] == crime]
    date_column = "Date"
    data.sort_values(by=date_column, inplace=True)
    data[date_column] = data[date_column].str[:10]
    wanted_columns = ["Latitude", "Longitude", "Date"]
    unwanted_columns = data.columns.difference(wanted_columns)
    data.drop(labels=unwanted_columns, axis="columns", inplace=True)
    print("Data was loaded successfully")
    return data


def append_min_max_and_step(
    data: pd.DataFrame, borders: dict[str, float], size: int, column: str
):
    """
    Calculate the minimum and maximum value for a column
    """
    minimum_key = f"minimum_{column}"
    maximum_key = f"maximum_{column}"
    step_key = f"{column}_step"
    minimum_value = data[column].min()
    maximum_value = data[column].max()
    borders[minimum_key] = minimum_value
    borders[maximum_key] = maximum_value
    borders[step_key] = (maximum_value - minimum_value) / size


def create_matrix_borders(data: pd.DataFrame, columns: list[str], size: int):
    """
    Generates the minimum and maximum values for a list of columns
    """
    borders = {}
    for column in columns:
        append_min_max_and_step(data, borders, size, column)
    return borders


def create_single_matrix(
    borders: dict[str, float], size: int, id_and_data: [str, pd.DataFrame]
) -> npt.NDArray:
    """
    Create a matrix
    """
    _, data = id_and_data
    matrix = np.zeros(shape=(size, size), dtype=int)
    if data.empty:
        return matrix
    start_longitude = borders["minimum_Longitude"]
    end_longitude = borders["minimum_Longitude"] + borders["Longitude_step"]
    latitude_column = "Latitude"
    longitude_column = "Longitude"

    for i in range(size):
        start_latitude = borders["minimum_Latitude"]
        end_latitude = borders["minimum_Latitude"] + borders["Latitude_step"]
        for j in range(size):
            crimes_in_day = data[
                (data[latitude_column] >= start_latitude)
                & (data[latitude_column] < end_latitude)
                & (data[longitude_column] >= start_longitude)
                & (data[longitude_column] < end_longitude)
            ]
            number_of_crimes, _ = crimes_in_day.shape
            matrix[i][j] = number_of_crimes
            start_latitude = end_latitude
            end_latitude += borders["Latitude_step"]
        start_longitude = end_longitude
        end_longitude += borders["Longitude_step"]
    return matrix


def create_squared_matrices(data: pd.DataFrame, borders: dict[str, float], size: int):
    """
    Create a list of matrices based on latitudes and longitudes
    """
    print("Creating matrices")
    date_column = "Date"
    matrices = []
    data_groups = data.groupby([date_column])
    with mp.Pool(5) as pool:
        matrices = list(
            tqdm(
                pool.imap_unordered(
                    partial(create_single_matrix, borders, size), data_groups
                ),
                total=len(data_groups),
            )
        )
    print("Matrices created successfully")
    return matrices


if __name__ == "__main__":
    CRIMES_PATH = "data/Crimes_-_2018.csv"
    CRIME_TARGET = "THEFT"
    MATRICES_SIZE = 64
    crime_data = load_dataframe(CRIMES_PATH, CRIME_TARGET)
    matrix_borders = create_matrix_borders(
        crime_data, ["Latitude", "Longitude"], MATRICES_SIZE
    )
    crime_matrices = create_squared_matrices(crime_data, matrix_borders, MATRICES_SIZE)
    with open("data/matrices.pickle", "wb") as handle:
        pickle.dump(crime_matrices, handle, protocol=pickle.HIGHEST_PROTOCOL)
