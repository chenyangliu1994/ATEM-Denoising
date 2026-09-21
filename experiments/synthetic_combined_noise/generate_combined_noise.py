import numpy as np
import os
from matplotlib import pyplot as plt
from SimPEG import maps
import SimPEG.electromagnetics.time_domain as tdem
from SimPEG.utils import plot_1d_layer_model
import random
from multiprocessing import Pool, cpu_count

def prog(output_file, iterations):
    plt.rcParams.update({"font.size": 16})

    # Source properties
    source_location = np.array([0.0, 0.0, 0])
    source_orientation = "z"
    source_current = 10.0
    source_radius = 6

    # Receiver properties
    receiver_location = np.array([0.0, 0.0, 0])
    receiver_orientation = "z"
    times = np.logspace(-5, -3, 40)

    receiver_list = [
        tdem.receivers.PointMagneticFluxDensity(
            receiver_location, times, orientation=receiver_orientation
        )
    ]
    waveform = tdem.sources.StepOffWaveform()

    source_list = [
        tdem.sources.CircularLoop(
            receiver_list=receiver_list,
            location=source_location,
            waveform=waveform,
            current=source_current,
            radius=source_radius,
        )
    ]
    survey = tdem.Survey(source_list)

    background_conductivity = 1e-1
    thicknesses = np.logspace(-0.4, 1.472, num=40, endpoint=True, base=10)
    n_layer = len(thicknesses) + 1

    with open(output_file, "a") as f:
        for _ in range(iterations):
            random_numbers = [random.randint(1, 1000)]
            for _ in range(39):
                if random.random() <= random.uniform(0.5, 1):
                    random_numbers.append(random_numbers[-1])
                else:
                    random_numbers.append(random.randint(1, 1000))
            random_numbers.append(random.randint(1, 1000))
            model = 1 / np.array(random_numbers)
            model_mapping = maps.IdentityMap(nP=n_layer)

            simulation = tdem.Simulation1DLayered(
                survey=survey,
                thicknesses=thicknesses,
                sigmaMap=model_mapping,
            )

            dpred = simulation.dpred(model)
            max_amplitude_20 = dpred[29]
            max_amplitude_10 = dpred[19]

            def generate_harmonic_noise(times, fixed_frequency=500):
                scale_factor = 10 ** (10 / 20)
                max_noise_amplitude = max_amplitude_20 * scale_factor
                harmonic_noise = np.sin(2 * np.pi * fixed_frequency * times) * max_noise_amplitude
                return harmonic_noise

            def generate_atmospheric_noise(data_length):
                num_spikes = np.random.randint(1, 6)
                spike_indices = np.random.choice(data_length, num_spikes, replace=False)
                atmospheric_noise = np.zeros(data_length)

                for idx in spike_indices:
                    spike_multiplier = np.random.uniform(1, 6)
                    spike_amplitude = abs(spike_multiplier * max_amplitude_10)
                    atmospheric_noise[idx] += spike_amplitude

                return atmospheric_noise

            def generate_gaussian_noise(data_length):
                return np.random.normal(0, max_amplitude_20, data_length)

            def generate_combined_noise(data_length):
                harmonic_noise = generate_harmonic_noise(np.linspace(0, 1, data_length))
                atmospheric_noise = generate_atmospheric_noise(data_length)
                gaussian_noise = generate_gaussian_noise(data_length)
                return harmonic_noise + atmospheric_noise + gaussian_noise

            noise = generate_combined_noise(len(dpred))
            dpred_with_noise = dpred + noise
            data_string = " ".join(map(str, dpred_with_noise)) + " " + " ".join(map(str, dpred))
            f.write(data_string + '\n')

def worker(proc_num, n, num_processes):
    output_file = f"tmp_data_out_{proc_num}.txt"
    iterations = n // num_processes
    prog(output_file, iterations)

def worker_args(args):
    return worker(*args)

if __name__ == '__main__':
    np.random.seed(347)

    n = 500000
    num_processes = cpu_count()
    num_processes = min(num_processes, 32)

    with Pool(num_processes) as pool:
        pool.map(worker_args, [(i, n, num_processes) for i in range(num_processes)])

    with open("zaosheng.txt", "w") as f_out:
        for i in range(num_processes):
            with open(f"tmp_data_out_{i}.txt", "r") as f_in:
                f_out.write(f_in.read())
            os.remove(f"tmp_data_out_{i}.txt")
