import numpy as np
import matplotlib.pyplot as plt

data = np.array([[ 0.23,  0.07],\
 [ 0.23, -0.03],\
 [ 0.23,  0.1 ],\
 [ 0.17, -0.09],\
 [ 0.17,  0.14],\
 [ 0.19, -0.2 ],\
 [ 0.17,  0.07],\
 [ 0.17,  0.1 ],\
 [ 0.24,  0.28],\
 [ 0.23, -0.18],\
 [ 0.18, -0.14],\
 [ 0.17, -0.02],\
 [ 0.18, -0.11],\
 [ 0.19,  0.25],\
 [ 0.23,  0.12],\
 [ 0.18,  0.17],\
 [ 0.24,  0.25],\
 [ 0.19,  0.21],\
 [ 0.23,  0.16],\
 [ 0.23, -0.11],\
 [ 0.17, -0.05],\
 [ 0.23, -0.16],\
 [ 0.17,  0.03],\
 [ 0.25,  0.34],\
 [ 0.23,  0.22],\
 [ 0.17,  0.05],\
 [ 0.18, -0.17],\
 [ 0.22, -0.  ]])

def poly2PointArray(coeffs,x1,x2, N = 20):
    x = np.linspace(x1,x2,N)
    y = np.zeros(x.shape)
    n = coeffs.shape[0]
    for lv1 in range(n):
        y = y + coeffs[n - lv1 - 1]* ( x ** lv1)
    return np.vstack((x,y)).T

def ransac_polyfit( x, y, deg=3, n=20, k=20, t=0.1, d=5, f=0.8):
    # n – minimum number of data points required to fit the model
    # k – maximum number of iterations allowed in the algorithm
    # t – threshold value to determine when a data point fits a model
    # d – number of close data points required to assert that a model fits well to data
    # f – fraction of close data points required
    
    besterr = 999999
    bestfit = None
    for kk in range(k):
        maybeinliers = np.random.randint(len(x), size=n)
        maybemodel = np.polyfit(x[maybeinliers], y[maybeinliers], deg)
        alsoinliers = np.abs(np.polyval(maybemodel, x)-y) < t
        if sum(alsoinliers) > d and sum(alsoinliers) > len(x)*f:
            bettermodel = np.polyfit(x[alsoinliers], y[alsoinliers], deg)
            thiserr = np.sum(np.abs(np.polyval(bettermodel, x[alsoinliers])-y[alsoinliers]))
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
    return bestfit
    
coeffs = np.array([2,5,3])
coeffs2 = np.array([2,3])

def errorTwoLanes( coeffs, wData, yData):
    m = coeffs[0]
    c = coeffs[1]
    L = 0.5

    if wData.shape[0] > 0:
        xWhite = wData[:,0]
        eWhite = wData[:,1] - (m*xWhite + c)

    if yData.shape[0] > 0:    
        xYellow = yData[:,0]
        eYellow = yData[:,1] - (m*xYellow + c + L*np.sqrt(m ** 2 + 1)/2)  

    if yData.shape[0] > 0 and wData.shape[0]>0:
        e = np.vstack((np.reshape(eWhite,(-1,1)), np.reshape(eYellow,(-1,1))))
    elif yData.shape[0]>0:
        e = np.reshape(eYellow,(-1,1))
    elif wData.shape[0]>0:
        e = np.reshape(eWhite,(-1,1))


    return e

def jacTwoLanes( coeffs, wData, yData):
    m = coeffs[0]
    c = coeffs[1]
    L = 0.5

    if wData.shape[0] > 0:
        xWhite = wData[:,0]
        jacWm = np.reshape(-xWhite,(-1,1))
        jacWc = np.reshape(-np.ones(xWhite.shape),(-1,1))
        jacW = np.hstack((jacWm, jacWc))

    if yData.shape[0] > 0:    
        xYellow = yData[:,0]
        jacYm = np.reshape(-xYellow + L*m/(2*np.sqrt(m**2 + 1)),(-1,1))
        jacYc = np.reshape(-np.ones(xYellow.shape),(-1,1))
        jacY = np.hstack((jacYm, jacYc))

    if yData.shape[0] > 0 and wData.shape[0]>0:
        jac = np.vstack((jacW, jacY))
    elif yData.shape[0]>0:
        jac = jacY
    elif wData.shape[0]>0:
        jac = jacW

    return jac

em = np.reshape(np.array([]),(-1,1))
print(np.vstack((em,em)))
e= errorTwoLanes(coeffs2, data, em)
jac = jacTwoLanes(coeffs2,em,data)

print(e)
print(jac)

arrayTest  = poly2PointArray(coeffs,0,10)

# Apparent fit, GARBAGE
m = 1.49
c = -0.25

x = np.linspace(-1,1,20)
y = m*x + c

# Test
z3 = ransac_polyfit(data[:,1],data[:,0],deg=3)
z1, res1, _, _, _ = np.polyfit(data[:,1],data[:,0],deg=3,full = True) # Least-squares fit straight line

z2, res2, _, _, _ = np.polyfit(data[:,1],data[:,0],deg=1,full = True) # Least-squares fit straight line

plotData = poly2PointArray(z2, -0.2 , 0.2)
y3 = (1/z2[0])*x - (z2[1]/z2[0])
plt.plot(data[:,1],data[:,0],'ro')
#plt.plot(x,y)
plt.plot(plotData[:,0],plotData[:,1])
#plt.plot(x,y3)
#plt.plot(arrayTest[:,0],arrayTest[:,1])
plt.axis('equal')
#plt.show()

"""
z1, res1, _, _, _ = np.polyfit(whitePointsArray[:,0],whitePointsArray[:,1],deg=1,full = True) # Least-squares fit straight line
z2, res2, _, _, _ = np.polyfit(whitePointsArray[:,1],whitePointsArray[:,0],deg=1,full = True) # Least-squares fit straight line
if res2 < res1:
    zWhite = np.array([1/z2[0], -(z2[1]/z2[0])])
else:
    zWhite = z1
"""