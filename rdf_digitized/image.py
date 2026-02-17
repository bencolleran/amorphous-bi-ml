import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
name="rdf_data_from_paper"
data=pd.read_csv(f'{PROJECT_ROOT}/rdf_digitized/{name}.csv',names=['r','g(r)'])
df=pd.DataFrame(data)
r=np.array(df['r'])
gr=np.array(df['g(r)'])

first=r[0]

idx = np.argsort(r)
r =r[idx]
gr= gr[idx]

r1=np.linspace(first,10,100)
pchip=PchipInterpolator(r,gr)
gr1=pchip(r1)

# Load image
img = mpimg.imread(f'{PROJECT_ROOT}/rdf_digitized/image.png')

# Load your data



fig, ax = plt.subplots(figsize=(6,4))

# Show image with correct data extents
ax.imshow(
    img,
    extent=[0, 10, 0, 3],   # x_min, x_max, y_min, y_max
    aspect='auto'
)

# Overlay the curve
ax.plot(r1, gr1, color="red", linewidth=2)

# Labels
ax.set_xlabel("r / Å")
ax.set_ylabel("g(r)")

plt.show()
plt.savefig(f'{PROJECT_ROOT}/rdf_digitized/{name}_with_image.png')

