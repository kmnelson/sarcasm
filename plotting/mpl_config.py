import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Set up global parameters for all plots
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 9
mpl.rcParams['ytick.major.size'] = 9
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['legend.borderaxespad'] = 2.0
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['xaxis.labellocation'] = 'right'
mpl.rcParams['yaxis.labellocation'] = 'top'

# For the 4 minor ticks between major ticks, we'll create a custom function
def setup_minor_ticks(ax):
    ax.margins(x=0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        
# Now when you create any plot, the rcParams will apply automatically
# You just need to call setup_minor_ticks() for each axis object
