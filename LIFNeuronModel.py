import torch
import time
import numpy as np
import MemristorModel as MM
import matplotlib.pyplot as plt
import Utilities as U
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LIF(MM.Memristor):
    def __init__(
        self,
        device_number = 1,
        dt = 1e-3, 
        Ron = 1e7, 
        Roff=1e12, 
        mu=220, 
        gamma1=600, 
        gamma2=2, 
        gamma3=200, 
        theta = 5,
        alpha=0.01, 
        beta=10, 
        lam=0.2,
        residue = True,
        random_init_res = False,
        symmetric = False,
        C2C = False,
        sigma1 = 1,
        sigma2 = 1,
        sigma3 = 1,
        D2D = False,
        D2D_level = 2,
        use_random_mu = False,
        use_random_beta = False,
        use_random_gamma1 = False,
        use_random_gamma2 = False,
        observe = False,
        Rin = 1e5,
        C = 10e-9,
        pulse_number = 1,
        low_period_pre = 50,    # us
        high_period = 5, # us
        low_period_post=0,
        amplitude=1
    ):
        super(LIF, self).__init__(dt=dt, Ron=Ron, Roff=Roff, mu=mu, gamma1=gamma1, gamma2=gamma2, gamma3=gamma3, theta= theta, alpha=alpha, 
                                                beta=beta, lam=lam, residue=residue, symmetric = symmetric, sigma1=sigma1, sigma2=sigma2, sigma3=sigma3,
                                                C2C=C2C, D2D=D2D, D2D_level=D2D_level, use_random_mu=use_random_mu, use_random_beta=use_random_beta, 
                                                use_random_gamma1=use_random_gamma1, use_random_gamma2=use_random_gamma2, observe=observe)

        self.Rin = torch.tensor(Rin)
        self.C = C
        self.pulse_number = pulse_number
        self.low_period_pre = (low_period_pre * 1e-6) // self.dt
        self.high_period = (high_period * 1e-6)  // self.dt
        self.low_period_post = (low_period_post * 1e-6)  // self.dt
        self.total_length = int(self.low_period_pre + self.high_period + self.low_period_post)
        self.device_number = device_number
        self.amplitude = amplitude
        self.random_init_res = random_init_res

    # the forward function is called each time we call the 1M1T1R neuron
    def forward(self):
        # Initializations
        self.D2D_sto( self.device_number)
        Rin = self.Rin.to(device)
        if self.random_init_res:
            res = torch.ones(1,  self.device_number).to(device) * 0.9
        else:
            res = torch.ones(1,  self.device_number).to(device) * 0.1
        fil = torch.zeros(1,  self.device_number).to(device)
        Vm = torch.zeros(1,  self.device_number).to(device)
        Rm = torch.ones(1,  self.device_number).to(device) * self.Roff
        S = torch.ones(1,  self.device_number).to(device)
        Rtot_rec = []
        Current_rec = []
        Vmem_rec = []

        if self.observe:
            Res_rec = []
            Fil_rec = []
            Vin_rec = []
            S_rec = []

        # Loop over time axis
        for i in range(self.pulse_number):
            for t in range(self.total_length):
                if t < self.low_period_pre:
                    vin = torch.zeros_like(Rm)
                elif self.low_period_pre <= t < self.low_period_pre + self.high_period:
                    vin = torch.ones_like(Rm) * self.amplitude
                else:
                    vin = torch.zeros_like(Rm)
                Iin = (vin - Vm) / Rin
                Im = Vm / Rm
                Ic = Iin - Im
                dVm = Ic / self.C * self.dt
                Vm = Vm + dVm
                res, fil, S, Rm = self.memristor_update(res, fil, S, Vm)


                if self.observe:
                    Res_rec.append(res)
                    Fil_rec.append(fil)
                    Vin_rec.append(vin)
                    S_rec.append(S)
                Rtot_rec.append(Rm)
                Current_rec.append(Im)
                Vmem_rec.append(Vm)
        
        if self.observe:
            self.rec = {
                'vin': torch.stack(Vin_rec,dim=1).detach().cpu().numpy(), #[batch,time_step,neuron]
                'cur': torch.stack(Current_rec,dim=1).detach().cpu().numpy(),
                'res': torch.stack(Res_rec,dim=1).detach().cpu().numpy(),
                'fil': torch.stack(Fil_rec,dim=1).detach().cpu().numpy(),
                'rtot': torch.stack(Rtot_rec,dim=1).detach().cpu().numpy(),
                'vmem': torch.stack(Vmem_rec,dim=1).detach().cpu().numpy(),
                'area': torch.stack(S_rec,dim=1).detach().cpu().numpy()
            }
        return torch.stack(Vmem_rec, dim=1).detach().cpu().numpy()
