import torch
import time
import numpy as np
import MemristorModel as MM
import LIFNeuronModel as LIF
import matplotlib.pyplot as plt
import Utilities as U
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def LIF_test():
    start_time  = time.time()
    # init a LIF model
    dt = 1e-6

    LIF_layer = LIF.LIF(dt = dt, 
                    Ron = 5e3, 
                    Roff=1e10, 
                    mu=150, 
                    gamma1=5000, 
                    gamma2=0.1, 
                    gamma3=10,
                    theta = 200,
                    alpha=0.01, 
                    beta=20,
                    lam=0.1,
                    residue = True,
                    random_init_res=True,
                    symmetric = True,
                    C2C = True,
                    sigma1 = 20, 
                    sigma2 = 20,
                    sigma3 = 0, 
                    D2D = False,
                    observe = True,
                    D2D_level = 0,
                    Rin = 4.7e4,
                    C = 4.7e-9,
                    pulse_number=50,
                    low_period_pre=35,
                    high_period=5,
                    amplitude=2.5
                    )

    s = LIF_layer.forward()
    device_number = LIF_layer.device_number
    time_step = s.shape[1]
    figure_path = f'Result/Figures/LIF_details.jpg'
    U.LIF_result_plot(s, LIF_layer.rec, figure_path, dt=dt)




def main():
    LIF_test()
    # LIF_freq_test()

if __name__ == "__main__":
    main()