#!/usr/bin/env python

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
import pstats

params = {'backend': 'ps',
          'axes.labelsize': 10,
          'text.fontsize': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': True,}
matplotlib.rcParams.update(params)


#Some simple functions to generate colours.
def pastel(colour, weight=2.4):
    """ Convert colour into a nice pastel shade"""
    rgb = np.asarray(colorConverter.to_rgb(colour))
    # scale colour
    #maxc = max(rgb)
    #if maxc < 1.0 and maxc > 0:
    #    # scale colour
    #    scale = 1.0 / maxc
    #    rgb = rgb * scale
    # now decrease saturation
    total = sum(rgb)
    slack = 0
    for x in rgb:
        slack += 1.0 - x

    # want to increase weight from total to weight
    # pick x s.t.  slack * x == weight - total
    # x = (weight - total) / slack
    x = (weight - total) / slack

    rgb = [c + 0.75*(x * (1.0-c)) for c in rgb]

    return rgb

def get_colours(n):
    """ Return n pastel colours. """
    base = np.asarray([[0.8,0.8,0], [0.8,0,0.8], [0,0.8,0.8]])

    if n <= 3:
        return base[0:n]

    # how many new colours do we need to insert between
    # red and green and between green and blue?
    needed = (((n - 3) + 1) / 2, (n - 3) / 2)
    
    colours = []
    for start in (0, 1):
        for x in np.linspace(0, 1-(1.0/(needed[start]+1)), needed[start]+1):
            colours.append((base[start] * (1.0 - x)) +
                           (base[start+1] * x))
    colours.append([0,0,1])

    return [pastel(c) for c in colours[0:n]]


time_components = {
                'CFL reduce' : "<method 'max' of 'petsc4py.PETSc.Vec' objects>",
                'Parallel initialization' : "<method 'create' of 'petsc4py.PETSc.DA' objects>",
                'Ghost cell communication' : "<method 'globalToLocal' of 'petsc4py.PETSc.DM' objects>",
                'Time evolution' : "evolve_to_time",
                'setup' : "setup"
}

def extract_profile(nsteps=200,ndim=3,solver_type='sharpclaw',nvals=(1,3,4),process_rank=0):

    stats_dir = './scaling'+str(nsteps)+'_'+str(ndim)+'d_'+str(solver_type)
    

    times = {}
    for key in time_components.iterkeys():
        times[key] = []
    times['Concurrent computations'] = []

    nprocs = []

    for n in nvals:
        num_processes = 2**(3*n)
        num_cells = 2**(6+n)
        nprocs.append(str(num_processes))
        prof_filename = os.path.join(stats_dir,'statst_'+str(num_processes)+'_'+str(num_cells)+'_'+str(process_rank))
        profile = pstats.Stats(prof_filename)
        prof = {}
        
        for key, value in profile.stats.iteritems():
            method = key[2]
            cumulative_time = value[3]
            prof[method] = cumulative_time
        for component, method in time_components.iteritems():
            times[component].append(round(prof[method],1))

    times['Concurrent computations'] = [  times['Time evolution'][i]
                                        - times['CFL reduce'][i]
                                        - times['Ghost cell communication'][i] 
                                        for i in range(len(times['Time evolution']))]

    return nprocs,times

def plot_and_table(nsteps=200,ndim=3,solver_type='sharpclaw',nvals=(1,3,4),process_rank=0):
    nprocs, times = extract_profile(nsteps,ndim,solver_type,nvals,process_rank)

    rows = ['Concurrent computations',
            'Parallel initialization',
            'Ghost cell communication',
            'CFL reduce']

    # Get some pastel shades for the colours
    colours = get_colours(len(rows))
    nrows = len(rows)

    x_bar = np.arange(len(nprocs)) + 0.3  # the x locations for the groups
    bar_width = 0.4
    yoff = np.array([0.0] * len(nprocs)) # the bottom values for stacked bar chart

    plt.axes([0.35, 0.25, 0.55, 0.35])   # leave room below the axes for the table

    for irow,row in enumerate(rows):
        plt.bar(x_bar, times[row], bar_width, bottom=yoff, color=colours[irow], linewidth=0)
        yoff = yoff + times[row]

    table_data = [times[row] for row in rows]

    # Add total efficiency to the table_data
    table_data.append( np.array([sum([row[i] for row in table_data]) for i in range(len(nprocs))]))
    table_data[-1] = table_data[-1][0]/table_data[-1]
    table_data[-1] = [round(x,2) for x in table_data[-1]]
    rows.append('Parallel efficiency')
    colours.append([1,1,1])

    # Add a table at the bottom of the axes
    mytable = plt.table(cellText=table_data,
                        rowLabels=rows, rowColours=colours,
                        colLabels=nprocs,
                        loc='bottom').set_fontsize(8)

    plt.ylabel('Execution Time for Process '+ str(process_rank)+' (s)')
    plt.figtext(.5, .02, "Number of Cores", fontsize=10)
    vals = np.arange(0, 36, 5)
    plt.yticks(vals, ['%d' % val for val in vals])
    plt.xticks([])
    plt.draw()

    f=plt.gcf()
    f.set_figheight(5) 
    f.set_figwidth(5) 

    plt.savefig('scaling_'+solver_type+'_'+str(ndim)+'D.pdf')

if __name__ == '__main__':
    plot_and_table()
