import scipy as sp
import pylab as plt
from scipy.integrate import odeint
import seaborn as sns
import math
import random
sns.set()

class HodgkinHuxley():
    """Full Hodgkin-Huxley Model implemented in Python"""
    C_m  =   1.0
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
    t = sp.arange(0.0, 1000.0, 0.01)
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

        
        I= 0.2*(t-100)*(t>100) - 0.2*100*(t>200) - 0.2*100*(t>300) - 0.2*100*(t>400)
           -0.2*100*(t>500) - 0.2*100*(t>600)-0.2*(t-100)*(t>700)+100*(t>700)
        I=10*(t>100) - 10*(t>200) + 10*(t>300) - 10*(t>400)+10*(t>500) - 10*(t>600)
        """
        I =  30*(t>300) - 30*(t>400) +30**(t>500) - 30*(t>600) 
    

        return I

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

        X = odeint(self.dALLdt, [-65, 0.05, 0.6, 0.32], self.t, args=(self,))
        V = X[:,0]
        m = X[:,1]
        h = X[:,2]
        n = X[:,3]
        ina = self.I_Na(V, m, h)
        ik = self.I_K(V, n)
        il = self.I_L(V)

        plt.figure()


        plt.subplot(2,1,1)
        plt.plot(self.t, m, 'r', label='m')
        plt.plot(self.t, h, 'g', label='h')
        plt.plot(self.t, n, 'b', label='n')
        plt.ylabel('Gating Value')
        
        plt.legend()

        plt.subplot(2,1,2)
        i_inj_values = [self.I_inj(t) for t in self.t]
        plt.plot(self.t, i_inj_values, 'k')
        plt.xlabel('t (ms)')
        plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
        plt.ylim(-10, 60)

        plt.show()

if __name__ == '__main__':
    runner = HodgkinHuxley()
    runner.Main()
