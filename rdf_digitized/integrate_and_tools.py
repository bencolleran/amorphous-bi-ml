import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from pathlib import Path
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def get(name):
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
    return [r1,gr1]

r1=get("rdf_data_from_paper")[0]
gr1=get("rdf_data_from_paper")[1]
r2=get("ovito_rdf")[0]
gr2=get("ovito_rdf")[1]
r3=get("liquid_rdf")[0]
gr3=get("liquid_rdf")[1]

def integrate(name):
    r=get(name)[0]
    gr=get(name)[1]
    rho=0.0273
    y=gr*r**2*rho*4*np.pi
    I=np.asarray([np.round(np.trapz(y[:i], r[:i]),1) for i in range(len(r))])
    #I=np.trapz(r,y)
    return I

#print(integrate("ovito_rdf")[40])

def maxima(name,var=None):
    r=get(name)[0]
    gr=get(name)[1]
    peaks, properties = find_peaks(gr,prominence=0.1)

    peak_r = r[peaks]
    peak_gr = gr[peaks]
    if var=="r":
        return peak_r
    elif var=="gr":
        return peak_gr
    else:
        return peak_gr,peak_r

#print(maxima("ovito_rdf",'r'))
print(maxima("rdf_data_from_paper","r"))

#to calculate density
s=maxima("rdf_data_from_paper","r")[0]/maxima("ovito_rdf",'r')[0]
print(s)



