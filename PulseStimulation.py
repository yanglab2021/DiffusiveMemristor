import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import MemristorModel as MM
import Utilities as U
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def AC_model_test(figure_path=['Result/Figures/Pulse_detail1.jpg', 'Result/Figures/Pulse_detail2.jpg'], data_save_path='Result/Data/simulation_data_pulse.pkl'):
    # init a memristor model
    dt = 1e-5
    time_step_num = int(1e-3//dt)
    neuron_num = 100
    batch_chosen = 0
    memristor = MM.Memristor(dt = dt,
                            Ron = 40e3,
                            Roff=1e9,
                            mu=100,
                            gamma1=800,
                            gamma2=10,
                            gamma3=4,
                            theta = 10,
                            alpha=0.01,
                            beta=80,
                            lam=0.1,
                            residue = True,
                            random_init_res=True,
                            symmetric = True,
                            C2C = True,
                            sigma1 = 10,
                            sigma2 = 0, 
                            sigma3 = 10, 
                            D2D = False,
                            observe = True,
                            D2D_level = 0.5,
                            use_random_mu = True,
                            use_random_beta = False,
                            use_random_gamma1 = False,
                            use_random_gamma2 = False,
                            R_Compliance = 22e3)
    # Generate input signal
    Vin = U.single_spike_test_signal(1, neuron_num, p1=int(1e-3//dt), p2=int(20e-3//dt), p3=int(30e-3//dt), p4=0*time_step_num, p5=0*time_step_num, p6=0*time_step_num)
 
    I = memristor.forward(Vin)
    # time_step_num=10
    U.AC_IV_plot(Vin[0, ::time_step_num].cpu().numpy(), I[0, ::time_step_num], figure_path[0], data_save_path, dt=dt*time_step_num)
    U.res_fil_r(I, memristor.rec, figure_path[1], 'Result/Data/AC_details.pkl', batch_chosen=batch_chosen, dt=dt, time_step_num=time_step_num)

def main():
    AC_model_test()

if __name__ == "__main__":
    main()