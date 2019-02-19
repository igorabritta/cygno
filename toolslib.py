import numpy as np
import glob, os
import re
import sys


def cluInfo(clusters,points,Ci,image,m_image,scale):
    
    labels = clusters.labels_
    pos    = points[labels == Ci]
    tag    = clusters.tag_[clusters.labels_== Ci][0]
    Xi     = list(pos[:,0].astype(int))
    Yi     = list(pos[:,1].astype(int))
    
    Is = []
    Ib = []
    ## to get X and Y in full dimension
    Xn = np.zeros(len(Xi)*scale*scale,dtype=int)
    Yn = np.zeros(len(Yi)*scale*scale,dtype=int)
    for i in range(0, len(Xi)):    # loop on single cluster value
        y0      = int(Xi[i]*scale)
        x0      = int(Yi[i]*scale)
        factor = scale*scale
        
        Yn[factor*i:factor*(i+1)] = np.reshape(np.reshape(np.arange((y0),(y0+scale)).repeat(scale,axis = 0),[scale,scale]).T,factor)
        Xn[factor*i:factor*(i+1)] = np.arange((x0),(x0+scale)).repeat(scale,axis = 0)
        
        Is.extend(list(image[(y0):(y0+scale),(x0):(x0+scale)].reshape(scale*scale).astype(int)))
        Ib.extend(list(m_image[(y0):(y0+scale),(x0):(x0+scale)].reshape(scale*scale).astype(int)))
    
    Xf = list(Xn.astype(int))
    Yf = list(Yn.astype(int))
    del Xi,Yi,Xn,Yn
      
    return Xf, Yf, Is, Ib, tag

        
def openTable(file):
    with open(file) as f:
        datain = eval(f.read())
    return datain

def saveTable(file,data):
    with open(file, 'w') as f:
        f.write(repr(data))
        

def colorbar(mappable):
    # plot colorbars        
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

def plot2hist(vari, bins = [20,40,100], liml = 0,limr = 50, label = '', scale = '', unity = '', density = True, logx = False, logy = False):
    ## Function to show in the same histogram the three categories
    
    # vari     = is a 3xN List which row has the information of one category       - List
    # nins     = is the number of bins to construct the histogram                  - int
    # liml     = Xlim min                                                          - float
    # limr     = Xlim max                                                          - float
    # label    = is the Xlabel to show in the plot                                 - string
    # scale    = is 'm' - mili, 'u' - micro, 'n' - nano,..                         - char
    # unity    = is the metric unit                                                - char
    # density  = is the flag to set the histogram to show the density or not       - Boolean
    # logx     = is the flag to set the X axis to log on the histogram or not      - Boolean
    # logy     = is the flag to set the Y axis to log on the histogram or not      - Boolean
    
    import matplotlib.pyplot as plt
    v = vari.copy()
    
    if scale == 'k':
        for i in range(0,3):
            v[i] = v[i]/(10**3)
    elif scale == 'M':
        for i in range(0,3):
            v[i] = v[i]/(10**6)
    elif scale == 'G':
        for i in range(0,3):
            v[i] = v[i]/(10**9)
    elif scale == 'T':
        for i in range(0,3):
            v[i] = v[i]/(10**12)
    elif scale == 'm':
        for i in range(0,3):
            v[i] = v[i]*(10**3)
    elif scale == 'u':
        for i in range(0,3):
            v[i] = v[i]*(10**6)
    elif scale == 'n':
        for i in range(0,3):
            v[i] = v[i]*(10**9)
    elif scale == 'p':
        for i in range(0,3):
            v[i] = v[i]*(10**12)
    else:
        for i in range(0,3):
            v[i] = v[i]
    
    
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)
    
    plt.hist(v[0], bins=bins[0], fc='r', alpha = 0.7, density=density)
    plt.hist(v[1], bins=bins[1], fc='b', alpha = 0.7, density=density)
    plt.hist(v[2], bins=bins[2], fc='darkorange', alpha = 0.7, density=density)
    plt.xlim([liml, limr])
    
    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")
    if density:
        plt.ylabel('Probability')
    else:
        plt.ylabel('Counts')
    plt.xlabel(label + '(' + scale + unity + ')',fontsize=18)
    plt.legend(['Recoils', 'Soft Electrons', 'MeV Electrons'],prop={'size': 18})
    plt.show()
    plt.close
    
def getTaggedVariable(vari,col):
    ## Function get the three categories on the same variable
    #    INPUT    
    # vari = is the DataFrame Pandas with all the information               - DataFrame
    # col  = is the name of the coloumn wanted                              - String
    #    OUTPUT
    # v = is a 3xN List which row has the information of one category       - List
    
    v = [np.array(vari[col][vari.Tag == 'l']), np.array(vari[col][vari.Tag == 'm']), np.array(vari[col][vari.Tag == 's'])]
    return v

def plotMesh(X,Y,Z,el,graus):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.view_init(elev=el, azim=graus)
    plt.show()
    return ax

#### Tools to rotate the cluster

def rotate(oX, oY, pX, pY, angle):
    from math import sin
    from math import cos
    from numpy import array
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox = oX
    oy = oY
    px = array(pX)
    py = array(pY)

    qx = ox + cos(angle) * (px - ox) - sin(angle) * (py - oy)
    qy = oy + sin(angle) * (px - ox) + cos(angle) * (py - oy)
    
    return qx.tolist(), qy.tolist()

def getAngle(X,Y):
    from numpy import polyfit
    from numpy import poly1d
    from numpy import arctan
    
    # - - - - Reta 0,0
    xo = [-2048,2048]
    yo = [0.001,0.001]
    
    zo = polyfit(yo,xo, 1)
    fo = poly1d(zo)    
    m1 = fo.c[0] 
    
    z = polyfit(X,Y, 1)
    func = poly1d(z) 
    m2 = func.c[0]
    
    
    
    angle = arctan(m1-m2/(1-m1*m2))
    
    return angle

def getC(X,Y):
    from numpy import polyfit
    from numpy import poly1d
    from numpy import arctan
    
    z = polyfit(X,Y, 1)
    func = poly1d(z)    
    
    return func.c[0]

def get_sliceleng(X,Y,pieces):
    # Function to get the mean length of the cluster
    # in X or Y direction.
    pieces = pieces
    
    newX = np.array(X) # Direction of the slices
    newY = np.array(Y) # Direction of the Mean Length
    
    slices = np.linspace(np.min(newX),np.max(newX),pieces)
    meanLY = np.zeros([(pieces-1),],dtype=float)

    for i in range(0,(pieces-1)):
    
        y = newY[(newX > slices[i]) & (newX < slices[i+1])]
        meanLY[i] = np.max(y) - np.min(y)
    return meanLY

def plot1hist(variable, bins, liml = 0, limr = 50, label = '', density = True, logx = False, logy = False):
    ## Function to show one variable on a histogram
    
    # variable = is a 1xN List with the variable information                       - List
    # nins     = is the number of bins to construct the histogram                  - int
    # nsd      = is the multiplication factor of the sigma to define the Xlim min  - float
    # nse      = is the multiplication factor of the sigma to define the Xlim max  - float
    # label    = is the Xlabel to show in the plot                                 - string
    # density  = is the flag to set the histogram to show the density or not       - Boolean
    # logx     = is the flag to set the X axis to log on the histogram or not      - Boolean
    # logy     = is the flag to set the Y axis to log on the histogram or not      - Boolean
    
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10,7))
    ax  = fig.add_subplot(111)
    
    v2  = variable
    e   = np.size(v2)
    m   = np.mean(v2[(v2 != 0) & (np.isnan(v2) == False)])
    s   = np.std(v2[(v2 != 0) & (np.isnan(v2) == False)])
    
    plt.hist(variable, bins=bins, fc='r', alpha = 0.7, density=density)
    plt.xlim([liml, limr])
    
    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")
    if density:
        plt.ylabel('Probability')
    else:
        plt.ylabel('Counts')
    plt.xlabel(label,fontsize=18)
    #plt.legend(['Recoils', 'Soft Electrons', 'MeV Electrons'],prop={'size': 18})

    textstr = '\n'.join((
        r'Entries $=%d$' % (e, ),
        r'Mean $=%.2f$' % (m, ),
        r'Std Dev$ =%.2f$' % (s, )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)

    # place a text box in upper left in axes coords
    plt.text(0.7, 0.9, textstr, fontsize=14,
            verticalalignment='top',transform=ax.transAxes, bbox=props)
    
    plt.show()

def pl3d(X,Y,Z,azim=0, bottom = 80):

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt



    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X, Y, Z, c='r', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim(bottom = bottom)
    ax.view_init(elev=0., azim=azim)

    plt.show()