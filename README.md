Please cite:

Jiawei Yang and Jeffrey M. Hausdorff. "Loss Analysis via Attention-scale for Physiologic Time Series". 2021.

#example

time_series=[1,2,3,4,5,3,1,3,0,4,2,1,3]

scale_factor=2 #int

scaling_model="PAS" # one of ["MS", "PAS", "OAS", "MAS"]

time_series_scaled=attention_scale(time_series, scale_factor, scaling_model)

#calculate Shonnon entropy

e=shannon_entropy(time_series_scaled)

print(e)
