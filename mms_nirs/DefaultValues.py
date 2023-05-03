from typing import List
import pandas as pd


class DefaultValues:
    def __init__(
        self,
        csv_file: str = "./defaults.csv",
        species: List[str] = ["HbO2", "HHb", "CCO"],
    ) -> None:
        df: pd.DataFrame = pd.read_csv(
            csv_file,
            sep=",",
            engine="pyarrow",
            index_col="wavelength",
        )

        self.extinction_coefficients = df[species].values
        self.wavelength_dependency = df["wl_dep"].values
        self.spectra_wavelengths = df.index.values
