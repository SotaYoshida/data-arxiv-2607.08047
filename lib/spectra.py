import numpy as np


def qpe_bins(sector, delta_t, ancillas):
    """Exact sector evolution followed by the inverse QFT, with little-endian output bins."""
    energies, vectors = np.linalg.eigh(np.array(sector["sector_hamiltonian_MeV"]))
    initial = sector["basis"].index(sector["initial_bitmask"])
    weights = np.abs(vectors[initial, :])**2
    shift = sector["energy_shift_MeV"]
    phases = -(energies - shift)*delta_t/(2*np.pi)
    if not np.all((0 <= phases) & (phases < 1)):
        raise ValueError("The shifted spectrum lies outside the chosen energy window.")
    size = 2**ancillas
    bins = np.arange(size)
    phases_in_time = np.exp(2j*np.pi*np.outer(bins, phases))
    amplitudes = np.fft.fft(phases_in_time, axis=0)/size
    probabilities = (np.abs(amplitudes)**2) @ weights
    return energies, shift - 2*np.pi*bins/(size*delta_t), probabilities
