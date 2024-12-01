import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import MemristorModel as MM
import Utilities as U
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def DC_model_test(memristor_type, data_save_path, figure_save_path):
    # init a memristor model
    cycle_num = 100
    device_num = 1
    
    dt = 1e-3
    time_step_num = int(1e-2//dt)

    if memristor_type == 'Sym':
        
        memristor = MM.Memristor(dt = dt, 
                            Ron = 1.7e5, 
                            Roff=1e11, 
                            mu=60, 
                            gamma1=60, 
                            gamma2=40, 
                            gamma3=10, 
                            theta = 100,
                            alpha=0.01, 
                            beta=40, 
                            lam=0.8,
                            symmetric = True,
                            C2C = True,
                            sigma1 = 8,
                            sigma2 = 5, 
                            sigma3 = 0, 
                            D2D = False,
                            observe = True,
                            D2D_level = 0.5,
                            use_random_mu = True,
                            use_random_beta = False,
                            use_random_gamma1 = False,
                            use_random_gamma2 = False,
                            I_Compliance = 1e-6)
        input_range = 0.5
        
    elif memristor_type =='Asym':
        memristor = MM.Memristor(dt = dt, 
                            Ron = 0.8e5, 
                            Roff=1e9, 
                            mu=100, 
                            gamma1=40, 
                            gamma2=40, 
                            gamma3= 5, 
                            theta = 100,
                            alpha=0.01, 
                            beta=40, 
                            lam=0.8,
                            residue = True,
                            symmetric = False,
                            C2C = True,
                            sigma1 = 8,
                            sigma2 = 5, 
                            sigma3 = 0, 
                            D2D = False,
                            observe = True,
                            D2D_level = 0.5,
                            use_random_mu = True,
                            use_random_beta = False,
                            use_random_gamma1 = False,
                            use_random_gamma2 = False,
                            I_Compliance = 1e-6)
        input_range = 0.2
    
    # Generate input signal
    Vin = U.generate_triangle_signal(device_num, time_step_num, cycle_num, range_min=1e-8, range_max=input_range, need_negative=True)
    I = memristor.forward(Vin)
    device_id = 0
    U.DC_IV_plot(Vin[device_id, ::time_step_num].cpu().numpy(), np.abs(I[device_id, ::time_step_num]), path=figure_save_path[0])
    U.res_fil_r(I, memristor.rec, path=figure_save_path[1], data_save_path=data_save_path, batch_chosen=device_id, dt=dt, time_step_num=time_step_num)
    

def main():
    Exp_path = 'Result/Data/experimental_data_symmetric.pkl'
    Sim_path = 'Result/Data/DC_simulation_data_symmetric.pkl'
    figure_save_path = ['Result/Figures/DC_sym_IV.jpg', 'Result/Figures/DC_sym_detail.jpg', 'Result/Figures/DC_sym_Vth.jpg', 'Result/Figures/DC_sym_Vhold.jpg']
    DC_model_test('Sym', Sim_path, figure_save_path[0:2])
    U.plot_Vth_Vhold(Exp_path=Exp_path, Simu_path=Sim_path, figure_save_path=figure_save_path[2:4])

    Exp_path = 'Result/Data/experimental_data_asymmetric.pkl'
    Sim_path = 'Result/Data/DC_simulation_data_asymmetric.pkl'
    figure_save_path = ['Result/Figures/DC_asym_IV.jpg', 'Result/Figures/DC_asym_detail.jpg', 'Result/Figures/DC_asym_Vth.jpg', 'Result/Figures/DC_asym_Vhold.jpg']
    DC_model_test('Asym', Sim_path, figure_save_path[0:2])
    U.plot_Vth_Vhold(Exp_path=Exp_path, Simu_path=Sim_path, figure_save_path=figure_save_path[2:4])

if __name__ == "__main__":
    main()