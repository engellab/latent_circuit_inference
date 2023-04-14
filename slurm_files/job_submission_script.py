import os
arguments = ["0.0066983_CDDM;relu;N=89;lmbdo=0.3;lmbdr=0.0;lr=0.002;maxiter=3000"]
for arg1 in arguments:
    print(arg1)
    os.system(f"sbatch LCI_gpu.slurm '{arg1}'")
