import pandas as pd

filepath = "/u/vld/sedm7085/project/rdf_digitized"

df = pd.read_csv(
    f"{filepath}/ovito_rdf.txt",
    sep=r"\s+",        # one or more spaces
    header=None        # no header row in your file
)

df.to_csv(
    f"{filepath}/ovito_rdf.csv",
    index=False,
    header=None
)
