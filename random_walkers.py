# Support functions

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from pandas import DataFrame, Series  # for convenience
import trackpy as tp

def generate_diff_track(pos0, D, dt, Tf):
    '''Function that generates a random walk from a given initial position, with
    chosen diffusivity
    INPUTS:
    pos0   : initial position (list of two elements X0 and Y0) [m]
    D      : diffusion coefficient [m2/s]
    dt     : time step for iteration [s]
    Tf     : final time [s]
    OUTPUTS:
    pos    : list of positions of the tracks, including pos0 [m]
    '''

    Nmax = int(np.floor(Tf/dt)) # number of timesteps given input
    pos = np.zeros((2,Nmax+1))  # initialisation of final position list
    
    pos[:,0] = pos0 # we know the first position from input
  
    # we compute a series of random displacements for each time point, along both axes
    rX = np.random.normal(0.0, np.sqrt(2*D*dt), Nmax)
    rY = np.random.normal(0.0, np.sqrt(2*D*dt), Nmax)
    
    # we then sum all these random displacements to obtain the final track
    pos[0,1:] = pos[0,0] + np.cumsum(rX)
    pos[1,1:] = pos[1,0] + np.cumsum(rY)
    
    return pos

def reflect(pos, L):
    '''Takes a random diffusive trajectory and return the trajectory
    with reflection against bounding box walls
    INPUTS
    pos     : 2xN array for X,Y position in time [m]
    L       : float, half width and half height of bounding box [m]
    OUTPUTS
    pos_box : 2xN array for X,Y position in time of the bound traj [m]'''

    i = 0
    while i<pos.shape[1]:
        if pos[0,i]>L:
            pos[0,i:] = pos[0,i:] + 2*L - 2*pos[0,i]
        elif pos[0,i]<-L:
            pos[0,i:] = pos[0,i:] -2*L - 2*pos[0,i]
        elif pos[1,i] > L:
            pos[1,i:] = pos[1,i:] + 2*L - 2*pos[1,i]
        elif pos[1,i]<-L:
            pos[1,i:] = pos[1,i:] -2*L - 2*pos[1,i]
        else:
            i +=1

    return pos

def plot_tracks_deltaX(N, D, Tf, ax_lim, deltaX_show, refresh, log_scale):
    '''Function plotting random tracks of particles and their associated displacements distribution.
    For this simulation, diffusivity is set.
    Designed to be used with ipywidgets
    INPUTS:
    N : number of particles
    D : diffusivity [um2/s]
    Tf: length of tracks [s]
    ax_lim: limit of axis display [um]
    deltaX_show: boolean for showing graphs of mean square displacements (MSDs)
    refresh: dummy argument for refreshing the widgets
    log_scale: boolean to show MSDs in loglog scale
    '''
    
    # we have a few constants
    dt  = 1e-2 # timestep of simulation [s]
    
    # we start all tracks from the center of the plot
    pos0 = [0, 0]
    
    # initialisation of figure
    fig, (ax, ax2) = plt.subplots(1,2,figsize = (12,6)) # option to use gridspec here
    
    deltaX1 = []
    deltaX2 = []
    deltaX3 = []
    for i in range(N):
        track = generate_diff_track(pos0, D*1e-12, dt, Tf) 
        track = track*1e6 # convert to microns
        
        # Now, we draw the trajectory and final position
        ax.plot(track[0,:],track[1,:], color = 'k', alpha = 0.5)
        ax.plot(track[0,-1],track[1,-1],marker='o', ms=6, markerfacecolor = 'm', markeredgecolor = 'k')
        
        # for displacements, we choose a few timestep multiples
        n1 = 1
        n2 = 3
        n3 = 5
        if i==0:
            deltaX1 = track[0,n1:]-track[0,:-n1]
            deltaX2 = track[0,n2:]-track[0,:-n2]
            deltaX3 = track[0,n3:]-track[0,:-n3]
        else:
            deltaX1 = np.concatenate((deltaX1, track[0,n1:]-track[0,:-n1]), axis=0)
            deltaX2 = np.concatenate((deltaX2, track[0,n2:]-track[0,:-n2]), axis=0)
            deltaX3 = np.concatenate((deltaX3, track[0,n3:]-track[0,:-n3]), axis=0)
        
    # mark initial position
    ax.plot(0,0,marker='o', ms=6, markerfacecolor = 'w', markeredgecolor = 'k')

    # polish the plot
    ax.set_xlabel(r'$x$ ($\mu$m)')
    ax.set_ylabel(r'$y$ ($\mu$m)')
    ax.axis('equal')
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    #ax.set_axis_off()
    
    if deltaX_show:
        
        maxX = np.max(np.abs(deltaX3))
        DeltaX = np.linspace(-maxX, maxX, 100)
        Nbins = 100

        # we plot the displacement distribution for a few time steps
        ax2.hist(deltaX1 , bins = Nbins, range = (-maxX, maxX), density = True, histtype = 'step', 
                 label = r'$\Delta t = {0} \times 10^{{-2}}$ s'.format(n1), log = log_scale)
        ax2.hist(deltaX2, bins = Nbins, range = (-maxX, maxX), density = True, histtype = 'step', 
                 label = r'$\Delta t = {0} \times 10^{{-2}}$ s'.format(n2), log = log_scale)
        ax2.hist(deltaX3, bins = Nbins, range = (-maxX, maxX), density = True, histtype = 'step', 
                 label = r'$\Delta t = {0} \times 10^{{-2}}$ s'.format(n3), log = log_scale)

        # we also plot the gaussian fit
        ax2.plot(DeltaX, np.maximum(np.exp(-DeltaX**2/(4*D*n1*dt))/(np.sqrt(4*np.pi*D*n1*dt)),10.0/(N*(Tf/dt)*2*maxX)),
                 linestyle = '--', color = 'k', alpha = 0.5, 
                 label =r'$\exp(-\Delta x^2/4 D \Delta t)/\sqrt{4 \pi D \Delta t}$')
        ax2.plot(DeltaX, np.maximum(np.exp(-DeltaX**2/(4*D*n2*dt))/(np.sqrt(4*np.pi*D*n2*dt)),10.0/(N*(Tf/dt)*2*maxX)),
                 linestyle = '--', color = 'k', alpha = 0.5)
        ax2.plot(DeltaX, np.maximum(np.exp(-DeltaX**2/(4*D*n3*dt))/(np.sqrt(4*np.pi*D*n3*dt)),10.0/(N*(Tf/dt)*2*maxX)),
                 linestyle = '--', color = 'k', alpha = 0.5)

        ax2.set(ylabel=r'probability distribution [$\mu$m$^-1$]',
        xlabel='displacement $\Delta x$ [$\mu$m]')
        ax2.set_xlim(-1.2, 1.2)
        
        plt.legend(loc=2, prop={'size': 10}, frameon = False)

    else:
        ax2.set_axis_off()
    
    
    
    fig.tight_layout()
    
    plt.show()

def plot_tracks_MSD(N, a, logmu, Tf, ax_lim, MSD_show, refresh, log_scale):
    '''Function plotting random tracks of particles and their associated mean square displacements,
    depending on several physical parameters and plotting options.
    Designed to be used with ipywidgets
    INPUTS:
    N : number of particles
    a : radius of particles [um]
    logmu: log of viscosity of fluid log[Pa s]
    Tf: length of tracks [s]
    ax_lim: limit of axis display [um]
    MSD_show: boolean for showing graphs of mean square displacements (MSDs)
    refresh: dummy argument for the widgets
    log_scale: boolean to show MSDs in loglog scale
    '''
    
    # we have a few constants
    dt  = 1e-2 # timestep of simulation [s]
    T   = 305  # temperature, fixed [K]
    k_B = 1.38e-23 # Boltzman constant [J K-1]
    
    # we compute the diffusivity [m2/s]
    D = k_B*T/(6*np.pi*10**(logmu)*a*1e-6) #
    
    # we start all tracks from the center of the plot
    pos0 = [0, 0]
    
    # initialisation of figure
    fig, (ax, ax2) = plt.subplots(1,2,figsize = (12,6)) # option to use gridspec here
    
    # we add a bullseye pattern for guiding the viewer
    angle = np.linspace( 0 , 2 * np.pi , 150 ) 
    x =  np.cos( angle ) 
    y =  np.sin( angle ) 
    r_list = np.arange(15) # in microns

    for i, r in enumerate(r_list):
        ax.plot(r*x, r*y, color = 'k', linestyle = '--', alpha = 0.5)
    
    pos = []
    for i in range(N):
        track = generate_diff_track(pos0, D, dt, Tf) 
        track = track*1e6 # convert to microns
        
        # Now, we draw the trajectory and final position
        ax.plot(track[0,:],track[1,:], color = 'k', alpha = 0.5)
        ax.plot(track[0,-1],track[1,-1],marker='o', ms=6, markerfacecolor = 'm', markeredgecolor = 'k')
        
        # we also populate a tracks panda frame (mostly empty) for use with trackpy
        track_pd = np.concatenate((track, np.zeros((6,track.shape[1])), 
                                   np.arange(0,track.shape[1],1)[np.newaxis,:],
                                   i*np.ones((1,track.shape[1]))),axis =0)
        if i==0:
            pos = track_pd
        else:
            pos = np.concatenate((pos, track_pd), axis=1)
        
    # mark initial position
    ax.plot(0,0,marker='o', ms=6, markerfacecolor = 'w', markeredgecolor = 'k')
    
        
    # we also plot the scaling
    ax.plot(np.sqrt(4*D*Tf)*1e6*x,np.sqrt(4*D*Tf)*1e6*y, color = 'r', linestyle = '--')
    ax.text(0.0, 1.3*np.sqrt(4*D*Tf)*1e6, r"$r = \sqrt{4 D t_\mathrm{f}}$", fontsize=16, color='r', 
            ha="center", va="center", bbox=dict(boxstyle="square", ec=(1.0, 1.0, 1.0), fc=(1., 1., 1.),alpha = 0.7))
  

    # polish the plot
    ax.set_xlabel(r'$x$ ($\mu$m)')
    ax.set_ylabel(r'$y$ ($\mu$m)')
    ax.axis('equal')
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    #ax.set_axis_off()
    
    if MSD_show:

        # we finish formatting the panda frame for use with Trackpy
        pos = np.transpose(pos)
        df = pd.DataFrame(pos, columns=['x', 'y','mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep', 'frame','particle'])

        # we compute the msd of each particle, and the ensemble value
        im = tp.imsd(df, mpp = 1., fps = 1/dt, max_lagtime=int(np.floor((Tf/dt)/2)))
        em = tp.emsd(df, mpp = 1., fps = 1/dt, max_lagtime=int(np.floor((Tf/dt)/2)))
        
        # we do a linear regression in log space on the whole range of times
        slope, intercept, r, p, stderr = stats.linregress(np.log(em.index), np.log(em))
        
        # we plot the individual msd with the theoretical trend and ensemble average overlaid
        ax2.plot(im.index, im, 'k-', alpha=0.2) 
        ax2.plot(em.index, 1e12*4*D*em.index, linestyle = '--', color='r', label=r'theoretical power law $\langle \Delta r^2 \rangle = 4 D \Delta t$')
        ax2.plot(em.index, em, 'o', label='ensemble average', alpha = 0.8)
        ax2.plot(em.index, np.exp(intercept)*em.index**slope,linestyle = '-.', color='m', label=r'fitted power law $\langle \Delta r^2 \rangle = \alpha \Delta t^\beta$')
        
        ax2.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
        xlabel='lag time $\Delta t$ [s]')
        if log_scale:
            ax2.set_xscale('log')
            ax2.set_yscale('log')
        plt.legend(loc=2, prop={'size': 12})
    else:
        ax2.set_axis_off()
    
    fig.tight_layout()
    
    if MSD_show:
    
        print(' ================================================================================================')
        print('| Theoretical Diffusivity D = {0:.2E} um2/s          |      Fitted power law: beta = {1:.2E}   |'.format(D*1e12,slope))
        print(' ================================================================================================')
    else:
        print(' =============================================')
        print('| Theoretical Diffusivity D = {:.2E} um2/s  |'.format(D*1e12))
        print(' =============================================')
    
    plt.show()

def plot_tracks_MSD_box(N, D, Tf, L, refresh, log_scale):
    '''Function plotting random tracks of particles in a square box and their associated mean square displacements,
    depending on several physical parameters and plotting options.
    Designed to be used with ipywidgets
    INPUTS:
    N : number of particles
    D : diffusivity of particles [um2/s]
    Tf: length of tracks [s]
    L: half-width of of bounding box [um]
    refresh: dummy argument for the widgets
    log_scale: boolean to show MSDs in loglog scale
    '''
    
    # we have a few constants
    dt  = 1e-2 # timestep of simulation [s]
    
    # we start all tracks from the center of the plot
    pos0 = [0, 0]
    
    # initialisation of figure
    fig, (ax, ax2) = plt.subplots(1,2,figsize = (12,6)) # option to use gridspec here
    
    # we add the box
    ax.plot([L,L,-L,-L,L], [-L,L,L,-L,-L], color = 'c', linestyle = '-.', alpha = 1,lw=2)
    
    pos = []
    for i in range(N):
        track = generate_diff_track([np.random.uniform(-L,L),np.random.uniform(-L,L)], D, dt, Tf)
        track_box = reflect(track,L)
        track_box = track_box*1e6 # convert to microns
        
        # Now, we draw the trajectory and final position
        ax.plot(track[0,:],track[1,:], color = 'k', alpha = 0.5)
        ax.plot(track[0,-1],track[1,-1],marker='o', ms=6, markerfacecolor = 'm', markeredgecolor = 'k')
        ax.plot(track[0,0],track[1,0],marker='o', ms=6, markerfacecolor = 'w', markeredgecolor = 'k')

        
        # we also populate a tracks panda frame (mostly empty) for use with trackpy
        track_pd = np.concatenate((track, np.zeros((6,track.shape[1])), 
                                   np.arange(0,track.shape[1],1)[np.newaxis,:],
                                   i*np.ones((1,track.shape[1]))),axis =0)
        if i==0:
            pos = track_pd
        else:
            pos = np.concatenate((pos, track_pd), axis=1)
        
    # mark initial position
    #ax.plot(0,0,marker='o', ms=6, markerfacecolor = 'w', markeredgecolor = 'k')

    # polish the plot
    ax.set_xlabel(r'$x$ ($\mu$m)')
    ax.set_ylabel(r'$y$ ($\mu$m)')
    ax.axis('equal')
    ax.set_xlim(-1.2*L, 1.2*L)
    ax.set_ylim(-1.2*L, 1.2*L)
    #ax.set_axis_off()

    # we finish formatting the panda frame for use with Trackpy
    pos = np.transpose(pos)
    df = pd.DataFrame(pos, columns=['x', 'y','mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep', 'frame','particle'])

    # we compute the msd of each particle, and the ensemble value
    im = tp.imsd(df, mpp = 1., fps = 1/dt, max_lagtime=int(np.floor((Tf/dt)/2)))
    em = tp.emsd(df, mpp = 1., fps = 1/dt, max_lagtime=int(np.floor((Tf/dt)/2)))
        
    # we plot the individual msd with the theoretical trend and ensemble average overlaid
    ax2.plot(im.index, im, 'k-', alpha=0.2) 
    ax2.plot(em.index, 4*D*em.index, linestyle = '--', color='r', label=r'theoretical power law $\langle \Delta r^2 \rangle = 4 D \Delta t$')
    ax2.plot(em.index, em, 'o', label='ensemble average', alpha = 0.8)
    ax2.plot(em.index, 0*em.index + 4*L**2/3,linestyle = '-.', color='c', label=r'box limit $4 L^2/3$')
        
    ax2.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
    xlabel='lag time $ \Delta t$ [s]')
    if log_scale:
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_ylim(2*D*dt, 10*L**2)
    else:
        ax2.set_ylim(0, 3*L**2)
    plt.legend(loc=2, prop={'size': 12})
    
    fig.tight_layout()
    
    plt.show()
    
    