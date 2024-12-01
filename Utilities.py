import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from datetime import datetime
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
import pickle
import scienceplots

def generate_triangle_signal(device_number, time_step_num, neuron_num, range_min=1e-8, range_max=0.6, need_negative=True, need_reset=True):
    # generate triangle signal
    Vin = torch.linspace(range_min, range_max, 201)
    Vin = torch.repeat_interleave(Vin, time_step_num)
    Vin_r = torch.linspace(range_max, range_min, 201)
    Vin_r = torch.repeat_interleave(Vin_r, time_step_num)
    Vin = torch.cat((Vin, Vin_r), dim=0)
    
    if need_negative:
        negative_Vin = - Vin
        if need_reset:
            Vin = torch.cat((Vin, torch.ones_like(Vin) * 1e-8), dim=0)
        Vin = torch.cat((Vin, negative_Vin), dim=0)

    Vin = Vin.repeat(device_number, neuron_num, 1).permute(0, 2, 1)
    return Vin

def single_spike_test_signal(batch_size, neuron_num, p1, p2, p3, p4, p5, p6, amplitude=1, read_voltage=0.05, reset_voltage=1e-6):
    period0 = torch.ones(1) * reset_voltage
    period1 = torch.ones(p1) * read_voltage
    period2 = torch.ones(p2) * amplitude
    period3 = torch.ones(p3) * read_voltage
    period4 = torch.ones(p4) * reset_voltage
    period5 = torch.ones(p5) * read_voltage
    period6 = torch.ones(p6) * reset_voltage

    single_pulse = torch.cat((period0, period1, period2, period3, period4, period5, period6), dim=0)

    return single_pulse.repeat(batch_size, neuron_num, 1).permute(0, 2, 1)

# DC static number extract
def Vth_Vhold_extract(I, V):
    I = np.abs(I)
    V = np.abs(V)
    Vth = []
    Vhold = []
    Vth_threshold = 1e-8
    Vhold_threshold = 1e-11
    for j in range(I.shape[1]):
        Vth_flag = 1
        Vhold_flag = 1
        I_pre_Vth = 0
        I_pre_Vhold = 0
        for i in range(1, I.shape[0]):
            if V[i, j] > V[i-1, j]:
                if I[i, j] > Vth_threshold and I_pre_Vth < Vth_threshold and Vth_flag:
                    Vth.append(V[i, j])
                    Vth_flag = 0
                I_pre_Vth = I[i, j]
            if V[i, j] < V[i-1, j]:
                if I[i, j] < Vhold_threshold and I_pre_Vhold > Vhold_threshold and Vhold_flag:
                    Vhold.append(V[i, j])
                    Vhold_flag = 0
                I_pre_Vhold = I[i, j]
    return np.abs(np.array(Vth)), np.abs(np.array(Vhold))

def plot_Vth_Vhold(Exp_path, Simu_path, figure_save_path):
    with open(Exp_path, 'rb') as file:
        exp_result = pickle.load(file)
    exp_Vth, exp_Vhold = Vth_Vhold_extract(exp_result['I'], exp_result['V'])
    with open(Simu_path, 'rb') as file:
        simu_result = pickle.load(file)
    simu_Vth, simu_Vhold = Vth_Vhold_extract(simu_result['I'], simu_result['V'])
    print(exp_result['I'].shape, exp_result['V'].shape, simu_result['I'].shape, simu_result['V'].shape)
    path_Vth = figure_save_path[0]
    path_Vhold = figure_save_path[1]
    
    plt.figure(figsize=(2.5, 2.2))

    plt.boxplot([exp_Vth, simu_Vth, exp_Vhold, simu_Vhold], labels=['Exp_Vth', 'Simu_Vth', 'Exp_Vhold', 'Simu_Vhold'])
    # plt.hist(exp_Vth, bins=30, alpha=0.5, label='Experiment')
    # plt.hist(simu_Vth, bins=30, alpha=0.5, label='Simulation')
    # plt.legend()
    plt.xticks(rotation=45)

    plt.ylabel('Voltage (V)', fontsize=10, labelpad=3)
    plt.tick_params(axis='y', labelsize=8, pad=3)
    plt.tick_params(axis='x', labelsize=8, pad=3)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.subplots_adjust(left=0.25, bottom=0.35, right=0.95, top=0.95)
    plt.savefig(path_Vth, dpi=300)


# DC plot
def res_fil_r(current, rec, path, data_save_path='DC_details.pkl', batch_chosen=0, dt=1e-7, time_step_num=1000):
    Vin = rec['vin']
    t_length = Vin.shape[1]
    sample_number = current.shape[2]
    t = np.linspace(0, t_length * dt, t_length)[::time_step_num]
    plt.figure(figsize=(5, 5))
    plt.subplots_adjust(hspace=0.15)
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.95, top=0.95)
    plt.subplot(5, 1, 1)
    plt.plot(t, Vin[batch_chosen, ::time_step_num, 0])
    plt.ylabel('Voltage (V)', fontsize=10, labelpad=3)
    plt.tick_params(axis='y', labelsize=8, pad=3)
    plt.xticks([])
    # plt.legend()
    plt.subplot(5, 1, 2)
    for i in range(sample_number):
        plt.plot(t, rec['rtot'][batch_chosen, ::time_step_num, i], label=f'cycle {i}')
    plt.ylabel('R (Ohm)', fontsize=10, labelpad=3)
    plt.yscale('log')
    plt.tick_params(axis='y', labelsize=8, pad=3)
    plt.xticks([])
    plt.subplot(5, 1, 3)
    for i in range(sample_number):
        plt.plot(t, rec['res'][batch_chosen, ::time_step_num, i] + rec['fil'][batch_chosen, ::time_step_num, i], label=f'cycle {i}')
    plt.ylabel('channel', fontsize=10, labelpad=3)
    plt.tick_params(axis='y', labelsize=8, pad=3)
    plt.xticks([])
    plt.subplot(5, 1, 4)
    for i in range(sample_number):
        plt.plot(t, rec['fil'][batch_chosen, ::time_step_num, i], label=f'cycle {i}')
    plt.ylabel('filament', fontsize=10, labelpad=3)
    plt.tick_params(axis='y', labelsize=8, pad=3)
    plt.xticks([])
    plt.subplot(5, 1, 5)
    for i in range(sample_number):
        plt.plot(t, rec['res'][batch_chosen, ::time_step_num, i], label=f'cycle {i}')
    plt.xlabel('time (s)', fontsize=10, labelpad=3)
    plt.ylabel('residue', fontsize=10, labelpad=3)
    plt.tick_params(axis='y', labelsize=8, pad=3)
    plt.tick_params(axis='x', labelsize=8, pad=3)
    
    plt.savefig(path, dpi=300)

    save_data = {'V': Vin[batch_chosen], 
                 'I': current[batch_chosen],
                 'T': t,
                 'R': rec['rtot'][batch_chosen], 
                 'Res': rec['res'][batch_chosen], 
                 'Fil': rec['fil'][batch_chosen],
                 }
    with open(data_save_path, 'wb') as file:
        pickle.dump(save_data, file)
    # plt.show()

def DC_IV_plot(V, I, path):
    plt.figure(figsize=(2.5, 2.5))
    for i in range(V.shape[1]):
        plt.plot(V[:, i], I[:, i])

    plt.xlabel('Voltage (V)', fontsize=10, labelpad=3)
    plt.ylabel('Current (A)', fontsize=10, labelpad=3)
    plt.tick_params(axis='y', labelsize=8, pad=3)
    plt.tick_params(axis='x', labelsize=8, pad=3)
    plt.ylim(1e-13, 1e-5)
    plt.yscale('log')
    if np.max(V) > 0.4:
        plt.xlim(-0.5, 0.5)
    plt.subplots_adjust(left=0.25, bottom=0.25)
    plt.savefig(path, dpi=300)


# AC plot
def AC_IV_plot(V, I, figure_path, data_save_path, dt=1e-7):
    t_length = V.shape[0]
    sample_number = I.shape[1]
    t = np.linspace(0, t_length * dt, t_length)*1e3
    plt.figure(figsize=(2.5, 2.5))
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(t, V[:, 0], 'b-', label='Vin', linewidth=0.5)
    ax1.set_ylabel('Voltage (V)', color='b', fontsize=10, labelpad=4)
    ax1.tick_params('y', colors='b')
    # 创建第二个Y轴
    ax2 = ax1.twinx()
    for i in range(sample_number):
    # 绘制VIn，左Y轴
        # 绘制current，右Y轴
        ax2.plot(t, I[:, i], label='Current', linewidth=0.5)
    ax2.set_ylabel('Current', color='r', fontsize=10, labelpad=4)
    ax2.tick_params('y', colors='r')

    ax1.tick_params(axis='y', labelsize=8, pad=3)
    ax2.tick_params(axis='y', labelsize=8, pad=3)
    # ax2.set_ylim(0, 1e-8)

    ax1.set_xlabel('time (ms)', fontsize=8, labelpad=3)
    ax1.tick_params(axis='x', labelsize=8, pad=3)
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8)
    # ax2.set_ylim(-1e-6, 1.2e-5)
    ax1.set_ylim(-0.02, 1.02)
    plt.savefig(figure_path, dpi=300)

    save_data = {'I': I, 'V': V, 'T': t}
    with open(data_save_path, 'wb') as file:
        pickle.dump(save_data, file)


# reservoir

def Reservoir_IV_plot(V, I, dt=1e-7):
    t_length = V.shape[1]
    sample_number = I.shape[0]
    t = np.linspace(0, t_length * dt, t_length)*1e3
    plt.figure(figsize=(10, 6))
    for i in range(3):
        ax1 = plt.subplot(3, 1, i+1)
        ax1.plot(t, V[0, :, i], 'b-', label='Vin')
        ax1.set_ylabel('Voltage (V)', color='b', fontsize=14, labelpad=4)
        ax1.tick_params('y', colors='b')
        # 创建第二个Y轴
        ax2 = ax1.twinx()
        for j in range(sample_number):
        # 绘制VIn，左Y轴
            # 绘制current，右Y轴
            ax2.plot(t, I[j, :, i]*1e6, label='Current')
        ax2.set_ylabel('Current (uA)', color='r', fontsize=14, labelpad=4)
        ax2.tick_params('y', colors='r')

        ax1.tick_params(axis='y', labelsize=14, pad=3)
        ax2.tick_params(axis='y', labelsize=14, pad=3)
        # ax2.set_ylim(0, 1e-8)

        ax1.set_xlabel('time (ms)', fontsize=14, labelpad=4)
        ax1.tick_params(axis='x', labelsize=14, pad=3)
        # ax2.set_ylim(-1e-6, 5e-5)
        # ax1.set_ylim(-0.02, 1.02)

def box_of_current(data):
    label = ['0001', '0010', '0011', '0100', '0101', '0110', '0111', '1000', 
             '1001', '1010', '1011', '1100', '1101', '1110', '1111', '0000', ]
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, vert=True, patch_artist=True)
    plt.xlabel('4-bit Patterns', fontsize=14)
    plt.ylabel('Current', fontsize=14)
    plt.yscale('log')
    plt.xticks(ticks=range(1, 17), labels=[i for i in label], fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    # plt.show()

def reservoir_res_fil_r(rec, batch_chosen=0, dt=1e-7, time_step_num=1000):
    Vin = rec['vin']
    t_length = Vin.shape[1]
    indices = [13, 8, 2]
    sample_number = len(indices)
    t = np.linspace(0, t_length * dt, t_length)[::time_step_num]
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(hspace=0.5)
    plt.subplot(5, 1, 1)
    for i in indices:
        plt.plot(t, Vin[batch_chosen, ::time_step_num, i])
    plt.ylabel('Voltage (V)', fontsize=12, labelpad=3)
    plt.tick_params(axis='y', labelsize=10, pad=3)
    plt.xticks([])
    plt.legend()
    plt.subplot(5, 1, 2)
    for i in indices:
        plt.plot(t, np.log10(rec['rtot'][batch_chosen, ::time_step_num, i]), label=f'cycle {i}')
    plt.ylabel('log(R)', fontsize=12, labelpad=3)
    plt.tick_params(axis='y', labelsize=10, pad=3)
    plt.xticks([])
    # plt.legend()
    plt.subplot(5, 1, 3)
    for i in indices:
        plt.plot(t, rec['res'][batch_chosen, ::time_step_num, i] + rec['fil'][batch_chosen, ::time_step_num, i], label=f'cycle {i}')
    plt.ylabel('channel', fontsize=12, labelpad=3)
    plt.tick_params(axis='y', labelsize=10, pad=3)
    plt.xticks([])
    plt.subplot(5, 1, 4)
    for i in indices:
        plt.plot(t, rec['fil'][batch_chosen, ::time_step_num, i], label=f'cycle {i}')
    plt.ylabel('filement', fontsize=12, labelpad=3)
    plt.tick_params(axis='y', labelsize=10, pad=3)
    plt.xticks([])
    plt.subplot(5, 1, 5)
    for i in indices:
        plt.plot(t, rec['res'][batch_chosen, ::time_step_num, i], label=f'cycle {i}')
    plt.xlabel('time (s)', fontsize=12, labelpad=3)
    plt.ylabel('residue', fontsize=12, labelpad=3)
    plt.tick_params(axis='y', labelsize=10, pad=3)
    plt.tick_params(axis='x', labelsize=10, pad=3)
    # plt.show()


# LIF
    
def OscillatoryNeuron_result_plot(current, rec, data_save_path, figure_path, batch_chosen=0, dt=1e-7, time_step_num=1):
    Vin = rec['vin']
    t_length = Vin.shape[1]
    t = np.linspace(0, t_length * dt, t_length)[::time_step_num]
    plt.figure(figsize=(5, 6.5))
    plt.subplots_adjust(hspace=0.15)
    plt.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.95)

    # for i in range(current.shape[2]):
    #     # plt.plot(t, rec['sens'][batch_chosen, ::time_step_num, i].detach().numpy() * (1e3), label='sensor')
    #     plt.plot(t, rec['vmem'][batch_chosen, ::time_step_num, i].detach().numpy(), label='mem')
    #     plt.plot(t, current[batch_chosen, ::time_step_num, i].repeat_interleave(3).detach().numpy(), label='spk')
    #     plt.plot(t, np.log10(rec['rm'][batch_chosen, ::time_step_num, i].detach().numpy())/12, label='R')
    # plt.xlabel('time (us)')
    # plt.ylabel('Current')
    # plt.legend()

    plt.subplot(6, 1, 1)
    for i in range(current.shape[2]):
        plt.plot(t, rec['vin'][batch_chosen, ::time_step_num, i], label='Input')
        plt.plot(t, rec['vmem'][batch_chosen, ::time_step_num, i], label='Membrane potential')
    plt.ylabel('Voltage (V)', fontsize=10, labelpad=3)
    plt.tick_params(axis='y', labelsize=8, pad=3)
    plt.xticks([])
    # plt.tick_params(axis='x', labelsize=8, pad=3)
    # plt.legend(frameon=False, fontsize='small')

    plt.subplot(6, 1, 2)
    for i in range(current.shape[2]):
        plt.plot(t, rec['cur'][batch_chosen, ::time_step_num, i]*1e6)
    # plt.yscale('log')
    plt.ylabel('Current (uA)', fontsize=10, labelpad=3)
    plt.xticks([])
    plt.tick_params(axis='y', labelsize=8, pad=3)

    plt.subplot(6, 1, 3)
    for i in range(current.shape[2]):
        plt.plot(t, rec['rtot'][batch_chosen, ::time_step_num, i])
    plt.yscale('log')
    plt.ylabel('log(R)', fontsize=10, labelpad=3)
    plt.xticks([])
    plt.tick_params(axis='y', labelsize=8, pad=3)

    plt.subplot(6, 1, 4)
    for i in range(current.shape[2]):
        plt.plot(t, rec['res'][batch_chosen, ::time_step_num, i] + rec['fil'][batch_chosen, ::time_step_num, i])
    
    plt.ylabel('channel', fontsize=10, labelpad=3)
    plt.tick_params(axis='y', labelsize=8, pad=3)
    plt.xticks([])

    plt.subplot(6, 1, 5)
    for i in range(current.shape[2]):
        plt.plot(t, rec['fil'][batch_chosen, ::time_step_num, i])
    # plt.xlabel('time (us)', fontsize=14, labelpad=20)
    plt.ylabel('filament', fontsize=10, labelpad=3)
    plt.tick_params(axis='y', labelsize=8, pad=3)
    plt.xticks([])

    plt.subplot(6, 1, 6)
    for i in range(current.shape[2]):
        plt.plot(t, rec['res'][batch_chosen, ::time_step_num, i])
    # plt.xlabel('time (us)', fontsize=14, labelpad=20)
    plt.ylabel('residue', fontsize=10, labelpad=3)
    plt.xlabel('time (s)', fontsize=10, labelpad=3)
    plt.tick_params(axis='y', labelsize=8, pad=3)
    plt.tick_params(axis='x', labelsize=8, pad=3)
    plt.savefig(figure_path, dpi=300)

    save_data = {'Vin': Vin[batch_chosen], 
                 'Vm': rec['vmem'][batch_chosen],
                 'T': t,
                 'R': rec['rtot'][batch_chosen], 
                 'Res': rec['res'][batch_chosen], 
                 'Fil': rec['fil'][batch_chosen],
                 }
    with open(data_save_path, 'wb') as file:
        pickle.dump(save_data, file)

    plt.show()
    
    
def LIF_result_plot(current, rec, figure_path, batch_chosen=0, dt=1e-7, time_step_num=1):
    Vin = rec['vin']
    t_length = Vin.shape[1]
    start_point = 700 # for zhongrui
    t_length = t_length - start_point
    t = np.linspace(0, t_length * dt, t_length)[::time_step_num] * 1e6
    plt.figure(figsize=(5, 6.5))
    plt.subplots_adjust(hspace=0.15)
    plt.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.95)

    plt.subplot(7, 1, 1)
    for i in range(current.shape[2]):
        plt.plot(t, rec['vin'][batch_chosen, start_point::time_step_num, i], label='Input')
        plt.plot(t, rec['vmem'][batch_chosen, start_point::time_step_num, i], label='Membrane potential')
    plt.ylabel('Voltage (V)', fontsize=10, labelpad=3)
    plt.tick_params(axis='y', labelsize=8, pad=3)
    plt.xticks([])
    plt.ylim(0, 0.4)
    # plt.tick_params(axis='x', labelsize=8, pad=3)
    # plt.legend(frameon=False, fontsize='small')

    plt.subplot(7, 1, 2)
    for i in range(current.shape[2]):
        plt.plot(t, rec['cur'][batch_chosen, start_point::time_step_num, i]*1e6)
    # plt.yscale('log')
    plt.ylabel('Current (uA)', fontsize=10, labelpad=3)
    plt.xticks([])
    plt.tick_params(axis='y', labelsize=8, pad=3)

    plt.subplot(7, 1, 3)
    for i in range(current.shape[2]):
        plt.plot(t, rec['rtot'][batch_chosen, start_point::time_step_num, i])
    plt.yscale('log')
    plt.ylabel('R (Ohm)', fontsize=10, labelpad=3)
    plt.xticks([])
    plt.tick_params(axis='y', labelsize=8, pad=3)

    plt.subplot(7, 1, 4)
    for i in range(current.shape[2]):
        plt.plot(t, rec['res'][batch_chosen, start_point::time_step_num, i] + rec['fil'][batch_chosen, start_point::time_step_num, i])
    
    plt.ylabel('Channel', fontsize=10, labelpad=3)
    plt.tick_params(axis='y', labelsize=8, pad=3)
    plt.xticks([])

    plt.subplot(7, 1, 5)
    for i in range(current.shape[2]):
        plt.plot(t, rec['fil'][batch_chosen, start_point::time_step_num, i])
    # plt.xlabel('time (us)', fontsize=14, labelpad=20)
    plt.ylabel('Filament', fontsize=10, labelpad=3)
    plt.tick_params(axis='y', labelsize=8, pad=3)
    plt.xticks([])

    plt.subplot(7, 1, 6)
    for i in range(current.shape[2]):
        plt.plot(t, rec['res'][batch_chosen, start_point::time_step_num, i])
    # plt.xlabel('time (us)', fontsize=14, labelpad=20)
    plt.ylabel('Residue', fontsize=10, labelpad=3)
    plt.tick_params(axis='y', labelsize=8, pad=3)
    plt.xticks([])

    plt.subplot(7, 1, 7)
    for i in range(current.shape[2]):
        plt.plot(t, rec['area'][batch_chosen, start_point::time_step_num, i])
    # plt.xlabel('time (us)', fontsize=14, labelpad=20)
    plt.ylabel('Area', fontsize=10, labelpad=3)
    plt.ylim(0, 2)
    plt.xlabel('Time (us)', fontsize=10, labelpad=3)
    plt.tick_params(axis='y', labelsize=8, pad=3)
    plt.tick_params(axis='x', labelsize=8, pad=3)
    plt.savefig(figure_path, dpi=300)
    plt.close()