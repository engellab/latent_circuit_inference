import os
arguments = ["CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0069041_20230401-135134"]
# arguments = ["CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0064612_20230403-160226",
#              "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0070679_20230403-201401",
#              "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0068963_20230403-202147",
#              "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0072453_20230403-201333",
#              "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0068146_20230403-175313"]#,
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0068341_20230403-201853",
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0066462_20230403-215733",
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0066983_20230403-220143",
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0069985_20230403-220506",
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0066581_20230403-202327",
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0068383_20230403-160527",
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.008884_20230403-215042",
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0067739_20230403-161144",
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0070495_20230403-161145",
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0066119_20230403-160411",
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0065229_20230403-160226",
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0069855_20230403-175047",
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0070615_20230403-202415",
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0073224_20230403-173740",
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0070738_20230403-220106",
             # "DM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0075082_20230403-220542",
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0070526_20230403-175047",
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0070948_20230403-161105",
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0068477_20230403-160143",
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0069759_20230403-160528",
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0069969_20230403-220035",
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0065435_20230403-175405",
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0073767_20230403-220506",
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0070801_20230403-220800",
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0069802_20230403-202226",
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0066214_20230403-201735",
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.007279_20230403-201333",
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0065982_20230403-161105",
             # "CDDM_relu;N=100;lmbdr=0.0;lmbdo=0.3_0.0066949_20230403-202118"]
for arg1 in arguments:
    print(arg1)
    os.system(f"sbatch LCI_gpu.slurm '{arg1}'")
