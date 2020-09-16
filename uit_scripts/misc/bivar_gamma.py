def corr_gamma(a1,a2,mu1,mu2,rho,N):
    """
    Use: X1,X2 = corr_gamma(a1,a2,b1,b2,rho,N)
    This function follows an algorithm obtained at
    http://web.ics.purdue.edu/~hwan/IE680/Lectures/Chap08Slides.pdf

    It generates the two gamma distributed random variables
    X1 ~ Gamma(a1,mu1)
    X2 ~ Gamma(a2,mu2)
    Corr(X1,X2) = rho, 0<=rho<=min(a1,a2)/sqrt(a1 a2)

    Input:
        a1: Shape parameter of X1 ................... float, a1>0
        a2: Shape parameter of X2 ................... float, a2>0
       mu1: Scale parameter of X1 ................... float, mu1>0
       mu2: Scale parameter of X2 ................... float, mu2>0
       rho: Correlation coefficient of X1 and X2. ... float, 0<=rho<=min(a1,a2)/sqrt(a1 a2)
         N: Number of RVs to generate. .............. int, N>0

    Output:
        X1: Gamma RV. ............................... (1xN) np array
        X2: Gamma RV. ............................... (1xN) np array
    """
    import numpy as np
    assert(a1>0)
    assert(a2>0)
    assert(mu1>0)
    assert(mu2>0)
    assert(rho>=0)

    rmax = min(a1,a2)/np.sqrt(a1*a2)
    if rho>rmax:
        rho=rmax
        print('Warning: r too large. Using maximal r:{}'.format(rho))

    rtmp = rho*np.sqrt(a1*a2)
    U = np.random.gamma(a1-rtmp,scale=1,size=N)
    V = np.random.gamma(a2-rtmp,scale=1,size=N)
    W = np.random.gamma(rtmp,scale=1,size=N)

    return mu1*(U+W),mu2*(V+W)

def bivar_gamma_mckay(a1,a2,mu,N):
    """
    Use: X1,X2 = bivar_gamma_mckay(a1,a2,b,r,N)
    This function gives the McKay bivariate gamma distribution,
    detailed on page 768 of doi: 10.1016/j.aml.2005.10.007.

    It generates the two gamma distributed random variables
    with (in general) different shape parameter and equal scale parameter.
    X1 ~ Gamma(a1,mu)
    X2 ~ Gamma(a2,mu)
    
    The pearson correlation coefficient is 
    rho = min(a1,a2)/sqrt(a1*a2)

    Input:
        a1: Shape parameter of X1 .................. float, a1>0
        a2: Shape parameter of X2 .................. float, a2>0
        mu: Common scale parameter of X1 and X2. ... float, mu>0
         N: Number of RVs to generate. ............. int, N>0

    Output:
        X1: Gamma RV. .............................. (1xN) np array
        X2: Gamma RV. .............................. (1xN) np array
    """
    import numpy as np
    assert(a1>0)
    assert(a2>0)
    assert(mu>0)
    
    a = max(a1,a2)
    b = a-min(a1,a2)
    
    U = np.random.beta(a,b,size=N)
    V = np.random.gamma(a+b,mu,size=N)

    return U*V,V

def bivar_gamma_1(a,mu1,mu2,rho,N):
    """
    Use: X1,X2 = bivar_gamma_1(a,mu1,mu2,rho,N)
    This function gives the bivariate gamma distribution from 
    assumption 1, Lemma 1, page 768 of doi: 10.1016/j.aml.2005.10.007.

    It generates the two gamma distributed random variables
    with equal shape parameter and (in general) different scale parameter.
    X1 ~ Gamma(a,mu1)
    X2 ~ Gamma(a,mu2)

    Input:
         a: Common shape parameter of X1 and X2. .... float, a>0
       mu1: Scale parameter of X1. .................. float, mu1>0
       mu2: Scale parameter of X2. .................. float, mu2>0
       rho: Correlation coefficient of X1 and X2. ... float, 0<rho<1
         N: Number of RVs to generate. .............. int, N>0

    Output:
        X1: Gamma RV. ............................... (1xN) np array
        X2: Gamma RV. ............................... (1xN) np array
    """
    import numpy as np
    assert(a>0)
    assert(mu1>0)
    assert(mu2>0)
    assert(rho>=0)
    
    # Neccesary condition
    if rho>0.5*np.sqrt(a/(1+a)):
        rho = 0.5*np.sqrt(a/(1+a))
        print('rho too large. rho set to its maximal value, {}'.format(rho))
    
    if rho==0:
        b=0
    else:
        par1 = 2+2*a-a/rho**2
        par2 = 1+2*a+a**2
        
        # We use the branch giving the smallest positive b.
        b = 0.5*(-par1 - np.sqrt(par1**2-4*par2))
        
        if b<0:
            b = 0.5*(-par1 + np.sqrt(par1**2-4*par2))

    W = np.random.beta(a,b,size=N)
    U = np.random.gamma(a+b,mu1,size=N)
    V = np.random.gamma(a+b,mu2,size=N)

    return U*W,V*W

def bivar_gamma_2(a1,a2,mu,rho,N):
    """
    Use: X1,X2 = bivar_gamma_1(a1,a2,mu,rho,N)
    This function gives the bivariate gamma distribution from 
    assumption 2, Lemma 1, page 768 of doi: 10.1016/j.aml.2005.10.007.

    It generates the two gamma distributed random variables
    with (in general) different shape parameter and equal scale parameter.
    X1 ~ Gamma(a1,mu)
    X2 ~ Gamma(a2,mu)

    Input:
        a1: Shape parameter of X1. .................. float, a1>0
        a2: Shape parameter of X2. .................. float, a2>0
        mu: Common scale parameter of X1 and X2. .... float, mu>0
       rho: Correlation coefficient of X1 and X2. ... float, 0<rho<1
         N: Number of RVs to generate. .............. int, N>0

    Output:
        X1: Gamma RV. ............................... (1xN) np array
        X2: Gamma RV. ............................... (1xN) np array
    """
    import numpy as np
    assert(a1>0)
    assert(a2>0)
    assert(mu>0)
    assert(rho>0)
    assert(rho<1)
    
    c = np.sqrt(a1*a2)/rho
    b1 = c-a1
    b2 = c-a2

    U = np.random.beta(a1,b1,size=N)
    V = np.random.beta(a2,b2,size=N)
    W = np.random.gamma(c,mu,size=N)

    return U*W,V*W
# EOF bivar_gamma.py
