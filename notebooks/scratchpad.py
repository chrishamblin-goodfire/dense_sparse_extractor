#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

#%%
# Light grey -> dark red colormap
cmap = LinearSegmentedColormap.from_list(
    "lightgrey_to_darkred",
    ["#f0f0f0", "#b22222"],  # lighter grey, lighter red
    N=256,
)

fig, ax = plt.subplots(figsize=(6, 1.2), constrained_layout=True)
ax.set_axis_off()

norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="horizontal")
cb.set_label("light grey → dark red")

plt.show()

# %%
