import scipy as sp
import pylab as plt
from scipy.integrate import odeint
import seaborn as sns
import math
sns.set()

class HodgkinHuxley():
    """Full Hodgkin-Huxley Model implemented in Python"""
    C_m  =   1.0,
    """membrane capacitance, in uF/cm^2"""
    g_Na = 120.0
    """Sodium (Na) maximum conductances, in mS/cm^2"""
    g_K  =  36.0
    """Postassium (K) maximum conductances, in mS/cm^2"""
    g_L  =   0.3
    """Leak maximum conductances, in mS/cm^2"""
    E_Na =  50.0
    """Sodium (Na) Nernst reversal potentials, in mV"""
    E_K  = -77.0
    """Postassium (K) Nernst reversal potentials, in mV"""
    E_L  = -54.387
    """Leak Nernst reversal potentials, in mV"""
    t = sp.arange(0.0, 1000.0, 0.5)
    """ The time to integrate over """
    def alpha_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.1*(V+40.0)/(1.0 - sp.exp(-(V+40.0) / 10.0))
    def beta_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 4.0*sp.exp(-(V+65.0) / 18.0)
    def alpha_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.07*sp.exp(-(V+65.0) / 20.0)
    def beta_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 1.0/(1.0 + sp.exp(-(V+35.0) / 10.0))
    def alpha_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.01*(V+55.0)/(1.0 - sp.exp(-(V+55.0) / 10.0))
    def beta_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.125*sp.exp(-(V+65) / 80.0)
##################################################################

    def inf_m(self, V):
        return self.alpha_m(V) / (self.alpha_m(V)+self.beta_m(V))
    def tau_m(self, V):
        return 1 / (self.alpha_m(V)+self.beta_m(V))
    def inf_h(self, V):
        return self.alpha_h(V) / (self.alpha_h(V)+self.beta_h(V))
    def tau_h(self, V):
        return 1 / (self.alpha_h(V)+self.beta_h(V))
    def inf_n(self, V):
        return self.alpha_n(V) / (self.alpha_n(V)+self.beta_n(V))
    def tau_n(self, V):
        return 1 / (self.alpha_n(V)+self.beta_n(V))
##################################################################
    def I_Na(self, V, m, h):
        """
        Membrane current (in uA/cm^2)
        Sodium (Na = element name)
        |  :param V:
        |  :param m:
        |  :param h:
        |  :return:
        """
        return self.g_Na * m**3 * h * (V - self.E_Na)
    def I_K(self, V, n):
        """
        Membrane current (in uA/cm^2)
        Potassium (K = element name)
        |  :param V:
        |  :param h:
        |  :return:
        """
        return self.g_K  * n**4 * (V - self.E_K)
    #  Leak
    def I_L(self, V):
        """
        Membrane current (in uA/cm^2)
        Leak

        |  :param V:
        |  :param h:
        |  :return:
        """
        return self.g_L * (V - self.E_L)
    def I_inj(self, t):
        """
        External Current
        |  :param t: time
        |  :return: step up to 10 uA/cm^2 at t>100
        |           step down to 0 uA/cm^2 at t>200
        |           step up to 35 uA/cm^2 at t>300
        |           step down to 0 uA/cm^2 at t>400   
        """
        return 10*(t>100) - 10*(t>200) + 35*(t>300) - 35*(t>800)
    @staticmethod
    def dALLdt(X, t, self):
        """
        Integrate
        |  :param X:
        |  :param t:
        |  :return: calculate membrane potential & activation variables
        """
        V, m, h, n = X
        dVdt = (self.I_inj(t) - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.C_m
        dmdt = self.alpha_m(V)*(1.0-m) - self.beta_m(V)*m
        dhdt = self.alpha_h(V)*(1.0-h) - self.beta_h(V)*h
        dndt = self.alpha_n(V)*(1.0-n) - self.beta_n(V)*n
        return dVdt, dmdt, dhdt, dndt
    def Main(self):
        """
        Main demo for the Hodgkin Huxley neuron model
        """
        X = odeint(self.dALLdt, [-100, 0.05, 0.6, 0.32], self.t, args=(self,))
        V = X[:,0]
        m = X[:,1]
        h = X[:,2]
        n = X[:,3]
      
        alpha_m = self.alpha_m(V)
        beta_m  = self.beta_m(V)
        alpha_h = self.alpha_h(V)
        beta_h  = self.beta_h(V)
        alpha_n = self.alpha_n(V)
        beta_n  = self.beta_n(V)

        inf_m   = self.inf_m(V)
        tau_m   = self.tau_m(V)
        inf_h   = self.inf_h(V)
        tau_h   = self.tau_h(V)
        inf_n   = self.inf_n(V)
        tau_n   = self.tau_n(V)

        plt.figure()

        plt.subplot(3,2,1)
        plt.plot(V, alpha_m, 'g',label='$\\alpha_m$')
        plt.plot(V, beta_m, 'b',label='$\\beta_m$')
        plt.ylabel('Rate ($ms^{-1}$)')
        plt.xlabel('$ V (mV)$')
        plt.legend()
        plt.ylim(-0.01, 30.0)
        plt.xlim(-100, 50)

        plt.subplot(3,2,2)
        plt.plot(V, inf_m, 'r',label='$\\inf_m$')
        plt.plot(V, tau_m, 'k',label='$\\tau_m$')
        plt.ylabel('')
        plt.xlabel('$V (mV)$')
        plt.legend()
        plt.ylim(-0.0, 1.01)
        plt.xlim(-100, 50)


        plt.subplot(3,2,3)
        plt.plot(V, alpha_h, 'g',label='$\\alpha_h$')
        plt.plot(V, beta_h, 'b',label='$\\beta_h$')
        plt.ylabel('Rate ($ms^{-1}$)')
        plt.xlabel('$V (mV)$')
        plt.legend()
        plt.ylim(-0.01, 1.0)
        plt.xlim(-100, 50)


        plt.subplot(3,2,4)
        plt.plot(V, inf_h, 'r',label='$\\inf_h$')
        plt.plot(V, tau_h, 'k',label='$\\tau_h$')
        plt.ylabel('')
        plt.xlabel('$V (mV)$')
        plt.legend()
        plt.ylim(-1, 10.01)
        plt.xlim(-100, 50)

        plt.subplot(3,2,5)
        plt.plot(V, alpha_n, 'g',label='$\\alpha_n$')
        plt.plot(V, beta_n, 'b',label='$\\beta_n$')
        plt.ylabel('Rate ($ms^{-1}$)')
        plt.xlabel('$V (mV)$')
        plt.legend()
        plt.ylim(-0.01, 1.0)
        plt.xlim(-100, 50)

        plt.subplot(3,2,6)
        plt.plot(V, inf_n, 'r',label='$\\inf_n$')
        plt.plot(V, tau_n, 'k',label='$\\tau_n$')
        plt.ylabel('')
        plt.xlabel('$V (mV)$')
        plt.legend()
        plt.ylim(-1, 6.01)
        plt.xlim(-100, 50)
        #plt.plot(1,1,1)
        #i_inj_values = [self.I_inj(t) for t in self.t]
        #plt.plot(self.t, i_inj_values, 'k')
        #plt.xlabel('t (ms)')
        #plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
        #plt.ylim(-1, 40)

        plt.show()
if __name__ == '__main__':
    runner = HodgkinHuxley()
    runner.Main()
