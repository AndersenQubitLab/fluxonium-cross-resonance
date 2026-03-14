from .autodiff import SampledFunction
from .square_spectrum import SquareSpectrum


class ESDOracle:
    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def interpolate(self, drive_frequency: float):
        self._check_drive_frequency(drive_frequency)
        return ESDOracleInterpolant(self.dataset, drive_frequency)

    def _check_drive_frequency(self, drive_frequency: float):
        domain = (
            self.dataset.drive_frequency.min(),
            self.dataset.drive_frequency.max(),
        )
        if drive_frequency < domain[0]:
            raise ValueError(
                f"{drive_frequency=} must be larger or equal to the dataset minimum"
                f" {self.dataset.drive_frequency.min().item()}"
            )
        if drive_frequency > domain[1]:
            raise ValueError(
                f"{drive_frequency=} must be smaller or equal to the dataset maximum"
                f" {self.dataset.drive_frequency.max().item()}"
            )


class ESDOracleInterpolant:
    def __init__(self, dataset, drive_frequency: float) -> None:
        self.dataset = dataset
        self.drive_frequency = drive_frequency
        self.drive_amplitude = dataset.amplitude.interp(
            drive_frequency=drive_frequency
        ).item()

    def get_matrix_element(self, harmonic: int, bra: int, ket: int) -> SquareSpectrum:
        ds = self.dataset\
                .sel(
                    harmonic=harmonic,
                    bra=bra,
                    ket=ket,
                ).interp(
                    drive_frequency=self.drive_frequency
                )

        frequency_points = (
            ds.pole.item()
            + ds.fourier_frequency.data
        )
        numerator_points = ds.numerator.data.ravel()

        numerator_func = SampledFunction(
            frequency_points,
            numerator_points,
        )
        spectrum_func = SquareSpectrum(
            numerator_func,
            pole=ds.pole.item(),
            bare_pole=ds.bare_pole.item(),
        )

        return spectrum_func
