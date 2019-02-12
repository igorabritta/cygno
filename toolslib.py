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

def plot2hist(variable,bins,nsd,nse,label, density = True, logx = False, logy = False):
    ## Function to show in the same histogram the three categories
    
    # variable = is a 3xN List which row has the information of one category       - List
    # nins     = is the number of bins to construct the histogram                  - int
    # nsd      = is the multiplication factor of the sigma to define the Xlim min  - float
    # nse      = is the multiplication factor of the sigma to define the Xlim max  - float
    # label    = is the Xlabel to show in the plot                                 - string
    # density  = is the flag to set the histogram to show the density or not       - Boolean
    # logx     = is the flag to set the X axis to log on the histogram or not      - Boolean
    # logy     = is the flag to set the Y axis to log on the histogram or not      - Boolean
    
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)
    
    v1 = np.append(variable[0],variable[1])
    v2 = np.append(v1,variable[2])
    
    m=np.mean(v2[(v2 != 0) & (np.isnan(v2) == False)])
    s=np.std(v2[(v2 != 0) & (np.isnan(v2) == False)])
    
    plt.hist(variable[0], bins=bins, range=(m-nsd*s, m+nse*s), fc='r', alpha = 0.7, density=density)
    plt.hist(variable[1], bins=bins, range=(m-nsd*s, m+nse*s), fc='b', alpha = 0.7, density=density)
    plt.hist(variable[2], bins=bins, range=(m-nsd*s, m+nse*s), fc='darkorange', alpha = 0.7, density=density)
    
    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")
    if density:
        plt.ylabel('Probability')
    else:
        plt.ylabel('Counts')
    plt.xlabel(label,fontsize=18)
    plt.legend(['Recoils', 'Soft Electrons', 'MeV Electrons'],prop={'size': 18})
    plt.show()
    
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