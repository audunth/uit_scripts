def yield_function(E,species='W'):
    """
    use: Y = yield_function(E,species='W')
    This method implements the modified Bohdansky formula for physical sputtering
    with incidence angle 0. See for example
    Y. Marandet et. al. PPCF 58 (2016) 114001

    input:
        E: Energy of incoming projetiles. ... (1xN) np array
        species: target particle species. ... 'Be' or 'W'. Default 'W'

    output:
        Y: Physical sputtering yield. ....... Same type and size as E
    """
    import numpy as np

    #Threshold energy in eV
    # Eth = EB/(gamma(1-gamma))
    # with EB the surface binding energy and 
    # gamma the maximum energy fraction transferred during collision.
    # gamma = 4 M_T M_P/(M_T+M_P)^2
    Eth = {'Be':13.09,'W':209.37}

    # Thomas-Fermi energy in eV
    ETF = {'Be':282,'W':9925}

    # Yield factor
    Q = {'Be':0.11,'W':0.07}

    # Nuclear stopping cross section
    def Sn(x):
        return 3.441*np.sqrt(x)*np.log(x+2.718) / ( 1+6.335*np.sqrt(x) + x*(6.882*np.sqrt(x) - 1.708 ))

    Y = np.zeros(E.size)
    good = E>=Eth[species]
    Y[good] =  Q[species]*Sn(E[good]/ETF[species])*(1-(Eth[species]/E[good])**(2/3))*(1-(Eth[species]/E[good]))**2
    return Y

def mean_yield(species='W',dist='normal',param=[0,1]):
    """
    use: mY = mean_yield(species='W',dist='normal',distparams=[0,1])

    Calculates the mean yield for a given species and a given distribution
    of the incoming energy.
    input:
        species: target particle species. .............. 'Be' or 'W'. Default 'W'
        dist: incoming particle energy distribution. ... 'normal' or 'gamma'. Default 'normal'
        param: list of distribution parameters. ........ if dist='normal': (mean,std)
                                                         if dist='gamma': (shape,scale)
    output:
        mY: mean of the Yield function (strictly <Y|E>Eth>)
    """
    import numpy as np
    from scipy.special import gamma
    from scipy.integrate import quad

    Eth = {'Be':13.09,'W':209.37}

    if dist=='normal':
        def pdf(x):
            return np.exp(-0.5*x**2)*(2.*np.pi)**(-0.5)

        def integrand(x):
            return pdf(x)*yield_function(param[1]*x+param[0],species)

        limit = (Eth[species]-param[0])/param[1]

        return quad(integrand,limit,np.inf)[0]

    elif dist=='gamma':
        mE = param[0]*param[1]
        rE = np.sqrt(param[0])*param[1]
        limit = (Eth[species]-mE)/rE
        def pdf(x):
            sp = np.sqrt(param[0])
            return sp**param[0] * (x+sp)**(param[0]-1) * np.exp(-sp*x-param[0]) / gamma(param[0])

        def integrand(x):
            return pdf(x)*yield_function(rE*x+mE,species)

        return quad(integrand,limit,np.inf)[0]
# EOF yield_function.py
