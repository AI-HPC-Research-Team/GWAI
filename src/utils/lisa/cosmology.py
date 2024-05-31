import math
from scipy import integrate
import scipy.optimize as optim
from .constant import Constant


class Cosmology(object):
    @staticmethod
    def H(zp, w):
        """
        Compute the Hubble parameter at redshift zp for a given w

        Args:
            zp (float): Redshift
            w (float): Dark energy equation of state parameter

        Returns:
            float: Hubble parameter
        """
        fn = 1.0 / (
            Constant.H0
            * math.sqrt(
                Constant.Omegam * math.pow(1.0 + zp, 3.0)
                + Constant.Omegalam * math.pow(1.0 + zp, 3.0 * w)
            )
        )
        return fn

    @staticmethod
    def DL(zup, w):
        """
        Compute the luminosity distance at redshift zup for a given w

        Args:
            zup (float): Redshift
            w (float): Dark energy equation of state parameter

        Returns:
            float: Luminosity distance
            float: Proper distance

        Usage: 
            DL(3,w=0)[0]
        """
        pd = integrate.quad(Cosmology.H, 0.0, zup, args=(w))[0]
        res = (1.0 + zup) * pd  # in Mpc
        return res * Constant.C_SI * 1.0e-3, pd * Constant.C_SI * 1.0e-3

    @staticmethod
    def findz(zm, dlum, ww):
        """
        Finding z for given DL, w
        
        Args:
            zm (float): Redshift
            dlum (float): Luminosity distance
            ww (float): Dark energy equation of state parameter

        Returns:
            float: Luminosity distance
        """
        dofzm = Cosmology.DL(zm, ww)
        return dlum - dofzm[0]

    @staticmethod
    def zofDl(DL, w, tolerance):
        """
        computes z(DL, w), Assumes DL in Mpc
        
        Args:
            DL (float): Luminosity distance
            w (float): Dark energy equation of state parameter
            tolerance (float): Tolerance

        Returns:
            float: Redshift
        """
        if tolerance > 1.0e-4:
            tolerance = 1.0e-4
        zguess = DL / 6.6e3
        zres = optim.fsolve(Cosmology.findz, zguess, args=(DL, 0.0), xtol=tolerance)
        return zres
