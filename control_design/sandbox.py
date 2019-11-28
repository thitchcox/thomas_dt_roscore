import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

x = np.linspace(0,2,40)


def f1(x):
    return 1.5*x ** 8

def f2(x):
    return 0.3*x ** 4 - 1 

def getMidTraj(f1,f2,startPoints, endPoints):
    # f1 and f2 are two function handles for the two trajectories
    # startPoints - two-element array containing the x-values of the start of each curve f1,f2
    # endPonts    - two-element array containing the x-values of the end of each curve f1,f2

    N = 20 # Discretize curve to get length
    x1 = np.linspace(startPoints[0], endPoints[0], N)
    y1sample = f1(x1)
    x2 = np.linspace(startPoints[1], endPoints[1], N)
    y2sample = f2(x2)

    # Build tables of x-coordinate vs length along curve
    Ltable1 = np.zeros((N,2))
    Ltable2 = np.zeros((N,2))
    for lv1 in range(1,x1.shape[0]):
        Ltable1[lv1][0] = x1[lv1]
        Ltable1[lv1][1] = np.sqrt((x1[lv1] - x1[lv1 - 1])**2 + (y1sample[lv1]-y1sample[lv1-1])**2) + Ltable1[lv1-1][1]
    for lv1 in range(1,x2.shape[0]):
        Ltable2[lv1][0] = x2[lv1]
        Ltable2[lv1][1] = np.sqrt((x2[lv1] - x2[lv1 - 1])**2 + (y2sample[lv1]-y2sample[lv1-1])**2) + Ltable2[lv1-1][1]

    # Total lengths of each curve
    L1 = Ltable1[-1][1]
    L2 = Ltable2[-1][1]

    # Now chop up by length
    N_len = 20
    l1 = np.linspace(0,L1,N_len)
    l2 = np.linspace(0,L2,N_len)

    # Query the x points corresponding to the lengths
    x1L = np.interp(l1,Ltable1[:,1],Ltable1[:,0])
    x2L = np.interp(l2,Ltable2[:,1],Ltable2[:,0])

    # Curve values at those points
    y1L = f1(x1L)
    y2L = f2(x2L)

    # Average is the middle trajectory
    xAvg = (x1L + x2L)/2
    yAvg = (y1L + y2L)/2

    return xAvg, yAvg


y1 = f1(x)
y2 = f2(x)

# Start and end points for both plots
xStart1 = 0
yStart1 = f1(xStart1)
xStart2 = 0
yStart2 = f2(xStart2)
xEnd1 = 0.5
yEnd1 = f1(xEnd1)
xEnd2 = 1.6
yEnd2 = f2(xEnd2)

# ###### Strategy 1 - only average the y values for each point in x
yAvgSimple = (f1(x)+f2(x))/2

# ###### Strategy 2 - Average both x and y by length
xAvg , yAvg = getMidTraj(f1,f2,np.array([xStart1,xStart2]),np.array([xEnd1,xEnd2]))


# ############ OLD - For plotting
# Get lengths of curve
x1 = np.linspace(xStart1, xEnd1, 10)
y1sample = f1(x1)
x2 = np.linspace(xStart2, xEnd2, 10)
y2sample = f2(x2)

Ltable1 = np.zeros((10,2))
Ltable2 = np.zeros((10,2))
for lv1 in range(1,x1.shape[0]):
    Ltable1[lv1][0] = x1[lv1]
    Ltable1[lv1][1] = np.sqrt((x1[lv1] - x1[lv1 - 1])**2 + (y1sample[lv1]-y1sample[lv1-1])**2) + Ltable1[lv1-1][1]
for lv1 in range(1,x2.shape[0]):
    Ltable2[lv1][0] = x2[lv1]
    Ltable2[lv1][1] = np.sqrt((x2[lv1] - x2[lv1 - 1])**2 + (y2sample[lv1]-y2sample[lv1-1])**2) + Ltable2[lv1-1][1]

L1 = Ltable1[-1][1]
L2 = Ltable2[-1][1]

# Now chop up by length
l1 = np.linspace(0,L1,10)
l2 = np.linspace(0,L2,10)

# Query the x points corresponding to the lengths
x1L = np.interp(l1,Ltable1[:,1],Ltable1[:,0])
x2L = np.interp(l2,Ltable2[:,1],Ltable2[:,0])

y1L = f1(x1L)
y2L = f2(x2L)


print(x1L)
print(x2L)

plt.plot(x,y1)
plt.plot(x,y2)
plt.plot([xStart1, xStart2, xEnd1, xEnd2],[yStart1, yStart2, yEnd1, yEnd2],'ro')
plt.plot(x,yAvgSimple)
plt.plot(xAvg,yAvg)
plt.plot(x1L, y1L,'ro')
plt.plot(x2L,y2L,'ro')
plt.ylim(-1, 1)
plt.xlim(0,2)
plt.show()
