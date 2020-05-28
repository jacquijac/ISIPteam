import matplotlib.pyplot as plt
import numpy as np
import math
import scipy

data=np.array([[ 382,  226],[ 387,  352],[ 464,  232],[ 499,  244],[ 423,  232],
       [ 577,  445], [ 337,  302],[ 315,  427],[ 286,  352],[ 446,  458],
       [ 478,  238],[ 429,  395],[ 527,  370],[ 510,  389],[ 460,  219],[ 348,  408],[ 393,  326],
       [ 410,  232],[ 377,  446],[ 474,  226],[ 513,  483],[ 367,  207],[ 455,  464],[ 443,  401],
       [ 399,  471],[ 445,  471],[ 344,  289]])

center=[452, 360]


xdat = data[:,0]
ydat = data[:,1]
ndat = len( xdat )

plt.plot(xdat, ydat, 'ro')


def fit_log_spiral(xdat, ydat, afit, bfit):

    dx = np.diff(xdat)
    dy = np.diff(ydat)

    dx_1=dx**2
    dy_1=dy**2
    sum_ar=dx_1+dy_1
    sqr_arr=sum_ar**0.5

    heading = np.unwrap(np.arctan2(dy,dx))
    dphi = np.diff( heading )
    #bapprox = np.arctan( dphi/2 )

    nda_1=ndat-2

    r = sqr_arr[0:nda_1]*np.tan(math.pi/2-dphi)

    xc = xdat[0:nda_1] + r * np.cos( heading[0:nda_1] + math.pi/2)
    yc = ydat[0:nda_1] + r * np.sin( heading[0:nda_1] + math.pi/2)

    xc1 = np.mean( xc )
    yc1 = np.mean( yc )
    return xc1, yc1


def ev_spi(center_real, xdat, ydat,ndat):
    xc = center_real[0]
    yc = center_real[1]
#% convert to r,theta
    x = xdat - xc
    y = ydat - yc
    r = ( x*x + y*y )**0.5
    theta = np.arctan2( y, x )
    theta = np.unwrap( theta )
#% linearized LSQ fit for r = a exp( b * theta )
#% ln(r) = ln(a) + b * theta
    p = np.polyfit( theta, np.log(r), 1 )
    afit = np.exp( p[1] )
    bfit = p[0]
#% evaluate fit
    rfit = afit * np.exp( bfit * theta );
    dr = r - rfit;
    dr_H = np.matrix(dr)
    get_H=dr_H.getH()
    rms = np.sqrt(abs( get_H*dr / ndat ))
    return rms, afit, bfit

rms, afit, bfit= ev_spi(center, xdat, ydat,ndat)

xc,yc=fit_log_spiral(xdat, ydat, afit, bfit)

minimum = scipy.optimize.fmin(func=ev_spi, x0=center, args=( xdat, ydat,ndat))
#xfit=minimum[0]
#yfit=minimum[1]
#x = xdat - xfit;
#y = ydat - yfit
#theta = np.arctan2( y/x )
#theta1 = np.unwrap( theta )
#th_max = max( theta1 )
#th_min = min( theta1 )
#dth = (th_max - th_min)/100
#th = np.arange(th_min, th_max, dth)
#rfit = afit * np.exp( bfit * theta1 )


#plt.plot(xdat, ydat, "ro",rfit*np.cos(theta1)+xfit, rfit*np.sin(theta1)+yfit,' g', c, d, 'b+'  )


