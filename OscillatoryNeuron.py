import torch
import time
import numpy as np
import MemristorModel as MM
import LIFNeuronModel as LIF
import matplotlib.pyplot as plt
import Utilities as U
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def OscillatoryNeuron_test():
    start_time  = time.time()
    # init a LIF model
    dt = 1e-6

    LIF_layer = LIF.LIF(dt = dt, 
                        Ron=5e3, 
                        Roff=1e8, 
                        mu=300, 
                        gamma1=200, 
                        gamma2=20, 
                        gamma3=10,
                        theta = 200,
                        alpha=0.01, 
                        beta=20, 
                        lam=1,
                        residue = True,
                        symmetric = True,
                        C2C = True,
                        sigma1 = 30,
                        sigma2 = 10, 
                        sigma3 = 0, 
                        D2D = False,
                        observe = True,
                        D2D_level = 0,
                        Rin = 150e3,
                        C = 6e-9,
                        pulse_number=1,
                        low_period_pre=50,
                        high_period=1e5,
                        low_period_post=50,
                        amplitude=1)

    s = LIF_layer.forward()
    device_number = LIF_layer.device_number
    time_step = s.shape[1]
    threshold = 0.3
    frequency = torch.zeros(device_number)
    for i in range(device_number):
        for j in range(5000, 10000):
            if s[0, j-1, i] < threshold and s[0, j, i] > threshold:
                frequency[i] += 1
    print(frequency / 0.5)
    data_save_path = 'Result/Data/simulation_OscillatoryNeuron_details_50k.pkl'
    figure_path = 'Result/Figures/OscillatoryNeuron_details.jpg'
    U.OscillatoryNeuron_result_plot(s, LIF_layer.rec, data_save_path, figure_path, dt=dt)
    
    
def OscillatoryNeuron_freq_test():
    dt = 1e-6
    Rin = torch.linspace(150e3, 250e3, 6)
    print(Rin)

    LIF_layer = LIF.LIF(dt = dt, 
                        Ron=5e3, 
                        Roff=1e8, 
                        mu=300, 
                        gamma1=200, 
                        gamma2=20, 
                        gamma3=10,
                        theta = 200,
                        alpha=0.01, 
                        beta=20, 
                        lam=1,
                        residue = True,
                        symmetric = True,
                        C2C = True,
                        sigma1 = 30,
                        sigma2 = 10, 
                        sigma3 = 0, 
                        D2D = False,
                        observe = True,
                        D2D_level = 0,
                        Rin = Rin,
                        C = 6e-9,
                        pulse_number=1,
                        low_period_pre=50,
                        high_period=1e5,
                        low_period_post=50,
                        amplitude=1)

    s = LIF_layer.forward()
    device_number = LIF_layer.device_number
    time_step = s.shape[1]
    threshold = 0.3
    frequency = torch.zeros(device_number)
    for i in range(device_number):
        for j in range(50000, 100000):
            if s[0, j-1, i] < threshold and s[0, j, i] > threshold:
                frequency[i] += 1
    frequency = frequency / 0.5
    print(frequency)

    # plt.plot(s[0, :, 0])
    # plt.show()
    
    plt.figure(figsize=(2.5, 2.5))
    plt.plot(Rin/1e3, frequency)
    plt.scatter(Rin/1e3, frequency)
    plt.ylabel('Frequency (Hz)', fontsize=10, labelpad=3)
    plt.xlabel('Resistance (kOhm)', fontsize=10, labelpad=3)
    plt.tick_params(axis='y', labelsize=8, pad=3)
    plt.tick_params(axis='x', labelsize=8, pad=3)
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.95, top=0.95)
    figure_path = 'Result/Figures/OscillatoryNeuron_R.jpg'
    plt.savefig(figure_path, dpi=300)




def main():
    OscillatoryNeuron_test()
    # OscillatoryNeuron_freq_test()

if __name__ == "__main__":
    main()