import numpy as np

def compute_h0(x0, x1, rho0=0.0001):
    # 1) intermediate a and b factors
    a = 95.0 * (0.04 - np.abs(x0))
    b = 95.0 * (1.0 - np.abs(x1))
    
    # 2) original dip calculation
    dip = 1.0 - 0.2495 * (1.0 + np.tanh(a)) * (1.0 + np.tanh(b))
    
    # 3) normalize dip into [0,1]
    normalized = (dip - 0.001014) / (0.999723 - 0.001014)
    normalized = np.clip(normalized, 0.0, 1.0)
    
    # 4) rescale so min = 0.0001, max = 1.0
    h0_min = rho0
    h0_max = 1.0
    h0 = h0_min + normalized * (h0_max - h0_min)
    
    return h0

def boudary(par1,par2,par3,par4,x_max=10,y_max=10,npoints_max=1000, x_min=.4, y_min=1.2, npoints_min=1000):

    xv_min = np.random.rand(npoints_min)*2*x_min-x_min
    yv_min = np.random.rand(npoints_min)*2*y_min-y_min
    xv_max = np.random.rand(npoints_max)*2*x_max-x_max
    yv_max = np.random.rand(npoints_max)*2*y_max-y_max
    x = np.concatenate([xv_max, xv_min])
    y = np.concatenate([yv_max, yv_min])
    p1 = np.ones_like(x)*par1
    p2 = np.ones_like(x)*par2
    p3 = np.ones_like(x)*par3
    p4 = np.ones_like(x)*par4
    t = np.zeros_like(x)

    parameters = np.stack([p1, p2, p3, p4, x, y, t], axis=1)

    h0=compute_h0(x, y, p4)

    data = np.zeros((npoints_min + npoints_max, 6))
    data[:,-1] = h0

    return parameters, data

if False:

    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3,1,figsize=(6,20))
    for i, p4 in enumerate([0.0001,0.3,0.5]):
        parameters,data=boudary(1,1,1,p4)

        sns.scatterplot(x=parameters[:,4],y=parameters[:,5],hue=data[:,-1],palette="viridis",s=1,ax=ax[i],hue_norm=(0,1))
    
    fig.savefig('plots/boundary_condition_test.png',dpi=300)
