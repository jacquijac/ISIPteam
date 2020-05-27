import matplotlib.pyplot as plt
import numpy as np
import math
data = np.array([
        [305 ,  406], [293, 401],
   [290,   383],[   300  , 365], [316,   356], [334,   358],
   [349,   369], [359, 390],[359, 421], [343,   438],
   [310,   454], [273,   450]
   
    ])


xdat = data[:,0]
ydat = data[:,1]
ndat = len( xdat )

dx = np.diff(xdat)
dy = np.diff(ydat)

dx_1=dx**2
dy_1=dy**2
sum_ar=dx_1+dy_1
sqr_arr=sum_ar**0.5

heading = np.unwrap(np.arctan2(dy,dx))
dphi = np.diff( heading )
bapprox = np.arctan( dphi/2 )

nda_1=ndat-2

r = sqr_arr[0:nda_1]*np.tan(math.pi/2-dphi)

xc = xdat[0:nda_1] + r * np.cos( heading[0:nda_1] + math.pi/2)
yc = ydat[0:nda_1] + r * np.sin( heading[0:nda_1] + math.pi/2)

xc = np.mean( xc )
yc = np.mean( yc )
cen = [ xc , yc ]



xc = cen[0]
yc = cen[1]
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
rms = ( dr*dr / ndat )**0.5



th_max = max( theta )
th_min = min( theta )
dth = ( th_max - th_min ) / 100;
th = np.arange(th_min, th_max, dth)
rfit = afit * np.exp( bfit * th )


plt.plot(xdat, ydat, "ro",rfit*np.cos(th)+cen[0], rfit*np.sin(th)+cen[1]  )





