#%% IMPORT LIBRARIES AND GENERAL SETTINGS
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#%% USER INPUT
# !!!!  FIRST, launch this script in the consolle: %matplotlib qt
station = 'sturla'

#%% 1. TEST TARGET PIC 
path_Target = "./" + station + "/images/Target/"
img_Target  = os.listdir(path_Target)

#%% 2. ASSESS PIXEL POSITIONS OF GCPs 
img = mpimg.imread(path_Target + img_Target[0])

fig, ax = plt.subplots()
ax.imshow(img)
ax.set_title("Click to select GCPs (z=zoom, c=click)")

# --- Variabili globali ---
mode = "click"
u, v = [], []
k = 1

# --- Callback click ---
def onclick(event):
    global mode, k, u, v
    if mode != "click":
        return

    if event.xdata is not None and event.ydata is not None:
        x, y = event.xdata, event.ydata
        print(f"Coordinate column,row (v,u) of pixel {k}: {x:.0f},{y:.0f}")
        ax.plot(x, y, 'ro', markersize=2)
        ax.text(x, y , str(k), color='r', fontsize=16, fontweight='bold')
        fig.canvas.draw()

        # Salva coordinate
        u.append(x)
        v.append(y)
        k += 1

# --- Callback tastiera ---
def onkey(event):
    global mode
    toolbar = plt.get_current_fig_manager().toolbar
    if event.key.lower() == 'z':
        toolbar.zoom()
        mode = "zoom"        
    elif event.key.lower() == 'c':
        toolbar.pan()
        toolbar.zoom()
        mode = "click"                

# --- Collega eventi ---
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', onkey)

plt.show()


