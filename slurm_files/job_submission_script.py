import os
arguments = ['0.0111542_CDDM;tanh;N=100;lmbdo=0.3;lmbdr=0.5;lr=0.005;maxiter=1500',
             '0.0112893_CDDM;tanh;N=100;lmbdo=0.3;lmbdr=0.5;lr=0.005;maxiter=1500',
             '0.0113189_CDDM;tanh;N=100;lmbdo=0.3;lmbdr=0.5;lr=0.005;maxiter=1500',
             '0.0113312_CDDM;tanh;N=100;lmbdo=0.3;lmbdr=0.5;lr=0.005;maxiter=1500',
             '0.0113474_CDDM;tanh;N=100;lmbdo=0.3;lmbdr=0.5;lr=0.005;maxiter=1500',
             '0.0113801_CDDM;tanh;N=100;lmbdo=0.3;lmbdr=0.5;lr=0.005;maxiter=1500',
             '0.0113808_CDDM;tanh;N=100;lmbdo=0.3;lmbdr=0.5;lr=0.005;maxiter=1500',
             '0.0114193_CDDM;tanh;N=100;lmbdo=0.3;lmbdr=0.5;lr=0.005;maxiter=1500',
             '0.0114748_CDDM;tanh;N=100;lmbdo=0.3;lmbdr=0.5;lr=0.005;maxiter=1500',
             '0.0114938_CDDM;tanh;N=100;lmbdo=0.3;lmbdr=0.5;lr=0.005;maxiter=1500',
             '0.0115308_CDDM;tanh;N=100;lmbdo=0.3;lmbdr=0.5;lr=0.005;maxiter=1500',
             '0.0115437_CDDM;tanh;N=100;lmbdo=0.3;lmbdr=0.5;lr=0.005;maxiter=1500',
             '0.0116138_CDDM;tanh;N=100;lmbdo=0.3;lmbdr=0.5;lr=0.005;maxiter=1500',
             '0.0116196_CDDM;tanh;N=100;lmbdo=0.3;lmbdr=0.5;lr=0.005;maxiter=1500',
             '0.0116338_CDDM;tanh;N=100;lmbdo=0.3;lmbdr=0.5;lr=0.005;maxiter=1500',
             '0.011634_CDDM;tanh;N=100;lmbdo=0.3;lmbdr=0.5;lr=0.005;maxiter=1500',
             '0.0116606_CDDM;tanh;N=100;lmbdo=0.3;lmbdr=0.5;lr=0.005;maxiter=1500',
             '0.0116637_CDDM;tanh;N=100;lmbdo=0.3;lmbdr=0.5;lr=0.005;maxiter=1500',
             '0.0116744_CDDM;tanh;N=100;lmbdo=0.3;lmbdr=0.5;lr=0.005;maxiter=1500',
             '0.0116865_CDDM;tanh;N=100;lmbdo=0.3;lmbdr=0.5;lr=0.005;maxiter=1500']
for arg1 in arguments:
    print(arg1)
    os.system(f"sbatch LCI_gpu.slurm '{arg1}'")
