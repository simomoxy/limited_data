import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RFFLayer(nn.Module):
    def __init__(self, kernel, F0, mc, in_dim, N_RF):
        super().__init__()
        
        self.F0 = F0
        self.mc = mc
        self.N_RF = torch.tensor(N_RF)
        
        self.ll_scale = nn.parameter.Parameter(
            torch.ones(1),
            requires_grad=False
        )
        #self.theta_llscale = nn.parameter.Parameter(
        #    torch.ones(((in_dim, N_RF))) * self.ll_scale,
        #    requires_grad=True
        #)
        
        self.log_scale = nn.parameter.Parameter(
            torch.ones(1)*0.5,
            requires_grad=True
        )
        
        self.omega_mu = nn.parameter.Parameter(
            torch.zeros((in_dim, N_RF)),
            requires_grad=True
        )
        self.omega_logsigma = nn.parameter.Parameter(
            (self.ll_scale) * torch.ones((in_dim, N_RF)),
            requires_grad=True
        )
        self.omega_eps = nn.parameter.Parameter(
            torch.randn((self.mc, in_dim, N_RF)),
            requires_grad=True
        )
        
        self.kernel_scale = 2 if kernel == 'arccos' else 1
        if kernel == 'rbf':
            self.kernel = lambda z: torch.cat([torch.cos(z), torch.sin(z)], dim=-1)
        elif kernel == 'arccos':
            self.kernel = lambda z: torch.maximum(z, torch.zeros(1, device=z.device))
    
    def get_prior(self):
        omega_mu_prior = torch.zeros(self.omega_mu.shape)
        omega_logsigma_prior = self.ll_scale.cpu() * torch.ones(self.omega_logsigma.shape).cpu()
        
        return omega_mu_prior, omega_logsigma_prior
        
    def forward(self, x):
        if self.F0:
            # x.shape = [bz, c, l] -> x.shape = [mc, bz, c, l]
            x = x.repeat(self.mc, 1, 1, 1)
            
        omega_var = torch.exp(self.omega_logsigma * 0.5)
        self.omega_q =  self.omega_mu + omega_var * self.omega_eps
        self.omega_q = torch.unsqueeze(self.omega_q, dim=1)
        phi_half =  torch.matmul(x.squeeze(), self.omega_q)
        #N_rf = torch.tensor(phi_half.shape[-1])
        
        scale = torch.exp(self.log_scale * 0.5)
        phi = self.kernel(phi_half)
        #print(phi)
        phi *= (self.kernel_scale * scale / torch.sqrt(self.N_RF))
        
        return phi
        

class RFFLinearLayer(nn.Module):
    def __init__(self, kernel, F0, mc, N_RF, out_dim):
        super().__init__()
        
        self.F0 = F0
        self.mc = mc
        N_RF = 2 * N_RF if kernel == 'rbf' else N_RF
        
        # scaler
        self.w_mu_prior = nn.Parameter(torch.zeros(1), requires_grad=False)
        # scaler
        self.w_logsigma_prior = nn.Parameter(torch.zeros(1), requires_grad=False)
        
        self.w_mu = nn.parameter.Parameter(
            torch.zeros((N_RF, out_dim)),
            requires_grad=True
        )
        self.w_logsigma = nn.parameter.Parameter(
            torch.ones((N_RF, out_dim)),
            requires_grad=True
        )
        self.w_eps = nn.parameter.Parameter(
            torch.randn((self.mc, N_RF, out_dim)),
            requires_grad=True
        )
        
    def forward(self, phi):
        w_var = torch.exp(self.w_logsigma * 0.5)
        W_q = self.w_mu + w_var * self.w_eps
        #print(phi.shape, W_q.shape)
        W_q = torch.unsqueeze(W_q, dim=1)
        F_y = torch.matmul(phi, W_q)
        
        return F_y
    

class ConvRFFLayer(nn.Module):
    def __init__(
        self,
        kernel,
        F0,
        mc,
        in_channels,
        out_channels, 
        kernel_size, 
        stride,
        padding,
        group=1,
    ):
        super(ConvRFFLayer, self).__init__()
        
        self.F0 = F0
        self.mc = mc
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.group = group
        
        self.ll_scale = nn.parameter.Parameter(
            torch.ones(1),
            requires_grad=False
        )
        
        self.log_scale = nn.parameter.Parameter(
            torch.ones(1)*0.5,
            requires_grad=True
        )
        
        self.omega_mu = nn.parameter.Parameter(
            torch.zeros((in_channels * kernel_size, out_channels)),
            requires_grad=True
        )
        self.omega_logsigma = nn.parameter.Parameter(
            (self.ll_scale) * torch.ones((in_channels * kernel_size, out_channels)),
            requires_grad=True
        )
        self.omega_eps = nn.parameter.Parameter(
            torch.randn((self.mc, in_channels * kernel_size, out_channels)),
            requires_grad=True
        )
        
        self.kernel_scale = 2 if kernel == 'rbf' else 1
        if kernel == 'rbf':
            self.kernel = lambda z: torch.cat([torch.cos(z), torch.sin(z)], dim=-1)
        elif kernel == 'arccos':
            self.kernel = lambda z: torch.maximum(z, torch.zeros(1, device=z.device))
    
        
    def get_prior(self):
        omega_mu_prior = torch.zeros(self.omega_mu.shape)
        omega_logsigma_prior = self.ll_scale.cpu() * torch.ones(self.omega_logsigma.shape).cpu()
        
        return omega_mu_prior, omega_logsigma_prior
    
    def forward(self, x):
        bz = x.shape[0]
        if self.F0:
            # x.shape = [bz, c, l] -> x.shape = [mc, bz, c, l]
            x = x.repeat(self.mc, 1, 1, 1)
            l = x.shape[-1]
            x = x.reshape(bz, self.mc * self.in_channels, l)
        
        omega_var = torch.exp(self.omega_logsigma * 0.5)
        self.omega_q =  self.omega_mu + omega_var * self.omega_eps
         
        #phi_half =  torch.matmul(x.squeeze(), self.omega_q)
        #N_rf = torch.tensor(phi_half.shape[-1])
        
        phi_half = conv1d(
            self.mc,
            x,
            self.omega_q,
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            group=self.group,
        )
        phi_half_size = phi_half.shape[-1]
        N_RF = torch.tensor(self.out_channels * phi_half_size)
        
        scale = torch.exp(self.log_scale * 0.5)
        phi = self.kernel(phi_half)
        #print(phi.shape, self.out_channels * phi_half_size)
        phi *= (self.kernel_scale * scale / torch.sqrt(N_RF))
        phi = phi.view(bz, self.kernel_scale * self.mc * self.out_channels, phi_half_size)
        #print(phi.shape)
        
        return phi
    

class ConvRFFLinearLayer(nn.Module):
    def __init__(
        self,
        kernel,
        F0,
        mc,
        in_channels,
        out_channels, 
        kernel_size, 
        stride,
        padding,
        group=1,
    ):
        super(ConvRFFLinearLayer, self).__init__()
        
        self.F0 = F0
        self.mc = mc
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.group = group
    
        self.kernel_scale = 2 if kernel == 'rbf' else 1
    
        # scaler
        self.w_mu_prior = nn.Parameter(torch.zeros(1), requires_grad=False)
        # scaler
        self.w_logsigma_prior = nn.Parameter(torch.zeros(1), requires_grad=False)
        
        self.w_mu = nn.parameter.Parameter(
            torch.zeros((self.kernel_scale * out_channels * kernel_size, out_channels)),
            requires_grad=True
        )
        self.w_logsigma = nn.parameter.Parameter(
            torch.ones((self.kernel_scale * out_channels * kernel_size, out_channels)),
            requires_grad=True
        )
        self.w_eps = nn.parameter.Parameter(
            torch.randn((self.mc, self.kernel_scale * out_channels * kernel_size, out_channels)),
            requires_grad=True
        )
        
    def forward(self, phi):
        w_var = torch.exp(self.w_logsigma * 0.5)
        W_q = self.w_mu + w_var * self.w_eps
        
        F_y = conv1d(
            self.mc,
            phi,
            W_q,
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            group=self.group,
            kernel_scale=self.kernel_scale,
        )
        
        return F_y


class RFFBlock(nn.Module):
    def __init__(self, kernel, F0, mc, in_dim, N_RF, out_dim):
        super().__init__()
        
        self.rff = RFFLayer(kernel, F0, mc, in_dim, N_RF)
        self.linear = RFFLinearLayer(kernel, F0, mc, N_RF, out_dim)
    
    def forward(self, x):
        #print(x.shape)
        phi = self.rff(x)
        #print(phi.sum())
        F_y = self.linear(phi)
        #print(F_y.sum())
        
        return F_y
    

class ConvRFFBlock(nn.Module):
    def __init__(
        self,
        kernel,
        F0,
        mc,
        in_channels,
        out_channels, 
        kernel_size, 
        stride,
        padding,
        group=1,
    ):
        super(ConvRFFBlock, self).__init__()
        
        self.conv_rff = ConvRFFLayer(kernel, F0, mc, in_channels, out_channels, kernel_size, stride, padding, group) 
        self.conv_linear = ConvRFFLinearLayer(kernel, F0, mc, out_channels, out_channels, kernel_size, stride, padding, group) 

    def forward(self, x):
        #print(x.shape)
        phi = self.conv_rff(x)
        #print(phi.sum())
        F_y = self.conv_linear(phi)
        #print(F_y.sum())
        
        return F_y
    

class RFFModel(nn.Module):
    def __init__(self,
                 kernel: str = 'rbf',
                 mc: int = 10,
                 num_layers: int = 1,
                 in_dims: int = 250,
                 N_RFs: list = [100],
                 num_classes: int = 1,
                 global_pool: bool=True,
                ):
        super().__init__()
        self.mc = mc
        self.global_pool = global_pool
        if not isinstance(N_RFs, list):
            N_RFs = [N_RFs] * num_layers
        self.final_N_RF = N_RFs[-1]
        
        blocks = []
        for i in range(num_layers):
            if i == 0:
                blocks.append(
                    RFFBlock(kernel, True, mc, in_dims, N_RFs[i], N_RFs[i])
                )
            else:
                blocks.append(
                    RFFBlock(kernel, False, mc, N_RFs[i-1], N_RFs[i], N_RFs[i])
                )
        self.blocks = nn.Sequential(*blocks)
        if self.global_pool:
            self.linear = nn.Linear(N_RFs[-1], num_classes)
        else:
            self.linear = nn.Linear(N_RFs[-1]*12, num_classes)

    def forward(self, x):
        bz = x.shape[0]
        x = self.blocks(x)
        if self.global_pool:
            x = x.reshape(self.mc, bz, -1, self.final_N_RF)
            x = x.mean(dim=2)
        else:
            x = x.reshape(self.mc, bz, -1)
        x = self.linear(x)
        
        return x
    
    def get_kl(self):
        kl = 0
        for mod in self.modules():
            if isinstance(mod, RFFLayer):
                omega_mu_prior, omega_logsigma_prior = mod.get_prior()
                omega_mu_prior, omega_logsigma_prior = omega_mu_prior.to(mod.omega_mu.device), omega_logsigma_prior.to(mod.omega_logsigma.device)
                kl += compute_kl(
                        mod.omega_mu, omega_mu_prior,
                        mod.omega_logsigma, omega_logsigma_prior
                    )
            elif isinstance(mod, RFFLinearLayer):
                kl += compute_kl(
                        mod.w_mu, mod.w_mu_prior,
                        mod.w_logsigma, mod.w_logsigma_prior
                )
        return kl
    
    def predict(self, x):
        x = self.forward(x)
        #x = torch.exp(x)
        x = torch.sigmoid(x)
        
        return x
    
    
class ConvRFFModel(nn.Module):
    def __init__(
        self,
        kernel: str='rbf',
        mc: int=10,
        num_layers: int=1,
        in_channels: int=1,
        feature_sizes: list=[64], 
        num_classes: int=1,
        length: int=250,
        global_pool: bool=True,
        kernel_size: int=3, 
        stride: int=2,
        padding: int=0,
        group: int=1,
    ):
        super(ConvRFF, self).__init__()
        
        self.mc = mc
        self.global_pool = global_pool
        if not isinstance(feature_sizes, list):
            feature_sizes = [feature_sizes] * num_layers
        self.final_feature = feature_sizes[-1]
        self.kernel_scale = 2 if kernel == 'rbf' else 1
        
        blocks = []
        for i in range(num_layers):
            if i == 0:
                blocks.append(
                    ConvRFFBlock(kernel, True, mc, in_channels, feature_sizes[i], kernel_size, stride, padding, group)
                )
            else:
                blocks.append(
                    ConvRFFBlock(kernel, False, mc, feature_sizes[i-1], feature_sizes[i], kernel_size, stride, padding, group)
                )
        self.blocks = nn.Sequential(*blocks)
        
        if global_pool:
            self.linear = RFFLinearLayer(kernel, False, mc, int(self.final_feature / self.kernel_scale), num_classes)
        else:
            length = calc_out_length(length, num_layers, kernel_size, stride, padding)
            self.linear = RFFLinearLayer(kernel, False, mc, int(length * self.final_feature / self.kernel_scale), num_classes)
    
    def forward(self, x):
        bz = x.shape[0]
        x = self.blocks(x)
        # add extra dim on pos 1 to be compatible with RFF class
        if self.global_pool:
            x = x.view(self.mc, 1, bz, self.final_feature, -1)
            x = x.mean(dim=-1)
        else:
            x = x.view(self.mc, 1, bz, -1)
        x = self.linear(x).squeeze()
        
        return x
        
    def get_kl(self):
        kl = 0
        for mod in self.modules():
            if isinstance(mod, ConvRFFLayer):
                omega_mu_prior, omega_logsigma_prior = mod.get_prior()
                omega_mu_prior, omega_logsigma_prior = omega_mu_prior.to(mod.omega_mu.device), omega_logsigma_prior.to(mod.omega_logsigma.device)
                kl += compute_kl(
                        mod.omega_mu, omega_mu_prior,
                        mod.omega_logsigma, omega_logsigma_prior
                    )
            elif isinstance(mod, ConvRFFLinearLayer):
                kl += compute_kl(
                        mod.w_mu, mod.w_mu_prior,
                        mod.w_logsigma, mod.w_logsigma_prior
                )
            elif isinstance(mod, RFFLinearLayer):
                kl += compute_kl(
                        mod.w_mu, mod.w_mu_prior,
                        mod.w_logsigma, mod.w_logsigma_prior
                )
        return kl
    
    def predict(self, x):
        x = self.forward(x)
        #x = torch.exp(x)
        x = torch.sigmoid(x)
        
        return x
        
            
def compute_kl(mu, mu_prior, logsigma, logsigma_prior):
    #log_sigma_prior.reshape(-1, 1)
    A = logsigma_prior - logsigma
    B = torch.pow(mu - mu_prior, 2) / torch.exp(logsigma_prior)
    C = torch.exp(logsigma - logsigma_prior) - 1
    kl = 0.5 * torch.sum(A + B + C)
    #kl = - 0.5 * torch.sum(1+ (logsigma*2) - mu.pow(2) - (logsigma*2).exp())
    
    return kl

def compute_ll(mc, out, y):
    y = y.repeat(mc, 1)
    sig = F.sigmoid(out)
    a = y*torch.log(sig)
    b = (1-y)*torch.log(1-sig)
    return (a + b).sum(dim=1) #-torch.sum(a + b, dim=1)


def calc_out_length(length, num_layers, kernel_size, stride, padding):
    for i in range(num_layers*2):
        length = ((length + 2 * padding - 1 * (kernel_size - 1) - 1) / stride) + 1
        length = math.floor(length)
        
    return length

def conv1d(mc, x, weight, in_channels, out_channels, kernel_size=3, stride=2, padding=0, group=1, kernel_scale=1):
    weight = weight.view(mc*out_channels, kernel_scale*in_channels, kernel_size)
    #print(x.shape, weight.shape)
    out = F.conv1d(x, weight, stride=stride, groups=mc*1, padding=padding)
    
    return out
