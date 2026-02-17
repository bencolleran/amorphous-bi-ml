import pandas as pd


df =pd.DataFrame(pd.read_csv("lattices.csv"))
#print(df.columns)
volumes=list(df["volume"])
print(max(volumes[1:]),(min(volumes)))
#806.35661871 86.770154423

max=19.289418662343614*16.705126585823212*26.15943795584872
min=16.850859319543172*14.593272246322146*22.852373966710704 
print(max,min)