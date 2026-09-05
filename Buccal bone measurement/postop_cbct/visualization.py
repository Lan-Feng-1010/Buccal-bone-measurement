"""Portable result figure; no local font file or private example dependency."""
from pathlib import Path
import numpy as np


def plot_measurement(result,section,output,raw=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
    x=np.array(result['slice']['x_mm']);y=np.array(result['slice']['y_mm'])
    fig,ax=plt.subplots(figsize=(7,5))
    image=section if raw is None else raw
    kwargs={}
    if raw is not None:
        finite=raw[np.isfinite(raw)]
        if len(finite):kwargs=dict(zip(['vmin','vmax'],np.percentile(finite,[2,98])))
    ax.imshow(image,cmap='gray',extent=[x[0],x[-1],y[-1],y[0]],interpolation='nearest',**kwargs)
    ax.axvline(0,color='#44B8D5',ls='--',lw=1)
    for row in result['measurements']:
        h=row['height_mm'];d=row['distance_mm']
        if d is not None:
            ax.plot([0,d],[h,h],color='#3BD1B2',lw=1.6)
            ax.scatter([0,d],[h,h],s=16,color='#3BD1B2')
        ax.text(-9,h,f'{h:g} mm',color='white',fontsize=9)
    ax.set(xlabel='Buccolingual coordinate (mm)',ylabel='Apical distance from supplied platform (mm)',
           title='Geometry output - requires anatomical and image review')
    fig.tight_layout()
    output=Path(output)
    for extension in ['png','pdf','svg']:fig.savefig(output/f'measurement.{extension}',dpi=300)
    plt.close(fig)
