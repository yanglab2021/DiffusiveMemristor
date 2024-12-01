import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Memristor(nn.Module):
    def __init__(
        self, 
        dt = 1e-3, 
        Ron = 1e7, 
        Roff=1e12, 
        mu=220, 
        gamma1=600, 
        gamma2=2, 
        gamma3 = 5,
        theta = 1,
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
        R_Compliance = 0,
        I_Compliance = 0
        ):
        super(Memristor, self).__init__()

        self.dt = dt
        self.Ron = Ron
        self.Roff= Roff
        self.mu= mu
        self.gamma1= gamma1
        self.gamma2= gamma2
        self.gamma3= gamma3
        self.alpha= alpha
        self.beta = beta
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.lam= lam
        self.theta = theta
        self.residue = residue
        self.symmetric = symmetric
        self.D2D = D2D
        self.C2C = C2C
        self.I_Compliance = I_Compliance
        self.R_Compliance = R_Compliance
        self.random_init_res = random_init_res


        if not self.residue:
            self.beta = torch.zeros_like(self.mu)

        # Stochasticity parameters
        self.D2D_level = D2D_level
        self.use_random_mu = use_random_mu
        self.use_random_beta = use_random_beta
        self.use_random_gamma1 = use_random_gamma1
        self.use_random_gamma2 = use_random_gamma2

        self.mu_lim = [mu*(1-self.D2D_level), mu*(1+self.D2D_level)]
        self.beta_lim = [beta*(1-self.D2D_level), beta*(1+self.D2D_level)]
        self.gamma1_lim = [mu*(1-self.D2D_level), mu*(1+self.D2D_level)]
        self.gamma2_lim = [beta*(1-self.D2D_level), beta*(1+self.D2D_level)]
        self.alpha_lim = [mu*(1-self.D2D_level), mu*(1+self.D2D_level)]

        # other auxiliary parameters
        self.observe = observe
        self.initialized = False
        self.para = {
            'mu': mu,
            'beta': beta,
            'gamma1': gamma1,
            'gamma2':  gamma2,
            'gamma3': gamma3,
            'sigma1': sigma1,
            'sigma2':  sigma2,
            'sigma3':  sigma3,
            'Ron': Ron,
            'Roff': Roff,
            'dt': dt,
            'D2D': self.D2D,
            'C2C': self.C2C,
            'residue': self.residue
        }

    def forward(self, Vin):
        # Initializations
        Vin = Vin.to(device)
        self.D2D_sto(Vin.shape[2])
        print(device)
        if self.random_init_res:
            res = torch.rand(Vin.shape[0], Vin.shape[2]).to(device) * 0.2 + 0.3
        else:
            res = torch.zeros(Vin.shape[0], Vin.shape[2]).to(device)
        fil = torch.zeros(Vin.shape[0], Vin.shape[2]).to(device)
        S = torch.zeros(Vin.shape[0], Vin.shape[2]).to(device)
        Rm = torch.ones(Vin.shape[0], Vin.shape[2]).to(device) * self.Roff
        Rtot_rec = []
        Current_rec = []

        if self.observe:
            Res_rec = []
            Fil_rec = []

        # Loop over time axis
        for t in range(Vin.shape[1]):
            # Vin = torch.abs(Vin)
            vin = Vin[:, t, :] / (self.R_Compliance + Rm) * Rm
            if self.I_Compliance:
                vin = torch.clamp(vin, max=self.I_Compliance * Rm, min = -Rm * self.I_Compliance)
            res, fil, S, Rm = self.memristor_update(res, fil, S, vin)

            if self.observe:
                Res_rec.append(res)
                Fil_rec.append(fil)
            Rtot_rec.append(Rm)
            current = Vin[:, t, :] / (Rm + self.R_Compliance)
            if self.I_Compliance:
                current = torch.clamp(current, max=self.I_Compliance, min=-self.I_Compliance)
            Current_rec.append(current)
        
        if self.observe:
            self.rec = {
                'vin': Vin.cpu().numpy(), #[batch,time_step,neuron]
                'res': torch.stack(Res_rec,dim=1).detach().cpu().numpy(),
                'fil': torch.stack(Fil_rec,dim=1).detach().cpu().numpy(),
                'rtot': torch.stack(Rtot_rec,dim=1).detach().cpu().numpy()
            }
        return torch.stack(Current_rec, dim=1).detach().cpu().numpy()
    
    def memristor_update(self, res, fil, S, vin):
        tot = (res + fil).clamp(0,1)
        mu = torch.ones_like(fil) * self.mu
        beta = torch.ones_like(fil) * self.beta
        gamma1 = torch.ones_like(fil) * self.gamma1
        gamma2 = torch.ones_like(fil) * self.gamma2
        gamma3 = torch.ones_like(fil) * self.gamma3
        theta = torch.ones_like(fil) * self.theta
        
        if self.C2C:
            gamma1 = torch.normal(gamma1, self.gamma1 * self.sigma1 * fil)
            gamma2 = torch.normal(gamma2, self.gamma2 * self.sigma2 * res)
            gamma3 = torch.normal(gamma3, self.gamma3 * self.sigma3)
            
        update_S_mask = ( tot.clone().detach() > 0.999999 ).float()
        updata_chanel_mask = ( S.clone().detach() < 1.000001 ).float()
        if self.symmetric:
            dS = (theta * torch.abs(vin) * torch.exp(-(S-1)) - gamma3 * (torch.exp(S-1))) * self.dt
            # dS = (theta * torch.abs(vin) * torch.exp(-(S-1)) - gamma3 * (S-1)) * self.dt
        else:
            dS = ( theta* vin.clamp(min=0) * torch.exp(1-S) - gamma3 * (torch.exp(S-1))) * self.dt
        
        S = (S + dS * update_S_mask).clamp(min=1)
            
        fil_diffuse = ((gamma1*fil)*self.dt).clamp(torch.zeros_like(fil), fil) * (updata_chanel_mask)
        fil = fil - fil_diffuse 
        res_diffuse = ((gamma2*res)*self.dt).clamp(torch.zeros_like(res), res) * (updata_chanel_mask)
        res = res - res_diffuse

        # drift = (vin * (mu / (self.alpha + (1 - tot)).clamp(self.alpha, self.alpha+1)) * self.dt).clamp(torch.zeros_like(fil), 1-fil-res)
        if self.symmetric:
            # drift = (torch.abs(vin) * (mu / (self.alpha + (1 - tot)).clamp(self.alpha, self.alpha+1)) * self.dt).clamp(torch.zeros_like(fil), torch.ones_like(1-fil-res))
            drift = (torch.abs(vin) * (mu / (self.alpha + (1 - tot)).clamp(self.alpha, self.alpha+1)) * self.dt).clamp(torch.zeros_like(fil), 1-fil-res) * (updata_chanel_mask)
        else:
            drift = (vin.clamp(min=0) * (mu / (self.alpha + (1 - tot)).clamp(self.alpha, self.alpha+1)) * self.dt).clamp(torch.zeros_like(fil), 1-fil-res) * (updata_chanel_mask)
            # drift = (vin.clamp(min=0) * (mu / (self.alpha + (1 - tot)).clamp(self.alpha, self.alpha+1)) * self.dt).clamp(torch.zeros_like(fil), torch.ones_like(1-fil-res))
        fil = fil + drift 
        transform = ((beta*fil)*self.dt).clamp(torch.zeros_like(fil), torch.min(1-res, fil)) * (updata_chanel_mask)
        # transform = ((beta*fil)*self.dt).clamp(torch.zeros_like(fil), fil)
        fil = fil - transform
        res = res + transform

        tot = (res + fil).clamp(0,1)
        Rm = self.Ron * tot / torch.pow(S, 2) + self.Roff * (torch.exp( (1-tot) / self.lam ) - 1) / ((torch.exp(torch.tensor( 1 / self.lam ) ))-1)
        return res, fil, S, Rm
    
        
    def D2D_sto(self, device_num):
        if self.D2D:
            if self.use_random_mu:
                self.mu = nn.Parameter(torch.Tensor(device_num))
                mean = (self.mu_lim[0] + self.mu_lim[1]) / 2
                nn.init.normal_(self.mu, mean=mean, std=mean*self.D2D_level)
                
            if self.use_random_beta:
                self.beta = nn.Parameter(torch.Tensor(device_num))
                mean = (self.beta_lim[0] + self.beta_lim[1]) / 2
                nn.init.normal_(self.beta, mean=mean, std=mean*self.D2D_level)
                
            if self.use_random_gamma1:
                self.gamma1 = nn.Parameter(torch.Tensor(device_num))
                mean = (self.gamma1_lim[0] + self.gamma1_lim[1]) / 2
                nn.init.normal_(self.gamma1, mean=mean, std=mean*self.D2D_level)

            if self.use_random_gamma2:
                self.gamma2 = nn.Parameter(torch.Tensor(device_num))
                mean = (self.gamma2_lim[0] + self.gamma2_lim[1]) / 2
                nn.init.normal_(self.gamma2, mean=mean, std=mean*self.D2D_level)
                
            if self.use_random_gamma3:
                self.gamma3 = nn.Parameter(torch.Tensor(device_num))
                mean = (self.gamma3_lim[0] + self.gamma3_lim[1]) / 2
                nn.init.normal_(self.gamma3, mean=mean, std=mean*self.D2D_level)


        