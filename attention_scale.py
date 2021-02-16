
"""

Reference: Jiawei Yang and Jeffrey M. Hausdorff. "Loss Analysis via Attention-scale for Physiologic Time Series". 2020.
Author: Jiawei YANG
Data: 2021.2.2
version: v1.0 rename variables.

"""


from collections import Counter
import numpy as np

def attention_scale(time_series, scale_factor, scaling_model):
    """
    Para:
        time_series: a one-dimensional discrete time series
        scale_factor: int (>0);
        scaling_model: int (-1 is mutiscale, 0 is peak-attention scale, 1 is frequency-attention-scale, 2 is median-attention-scale);
    Return: the coarse-grained time-series
    """

    # use all observations as attention observations
    def index_t_th(time_series):
        return [i for i in range(-1,len(time_series))]

    # find peak points as attention observations
    def index_peak_points(time_series):
        r = 0
        index = [i for i in range(len(time_series)) if
                 (i == 0) or (i == len(time_series) - 1) or (
                             time_series[i] - time_series[i + 1] > r and time_series[i] - time_series[i - 1]) > r or (
                         r > time_series[i] - time_series[i + 1] and r > time_series[i] - time_series[i - 1])]
        return index

    # find the items with most occurrences as attention observations
    def index_most_occurrences(time_series):
        occurence_count = Counter(time_series)
        most_occurrences_value = occurence_count.most_common(1)[0][0]
        index = [i for i in range(len(time_series)) if
                 i == 0 or i == len(time_series) - 1 or time_series[i] == most_occurrences_value]
        return index

    # find the items with median value as attention observations
    def index_median(time_series):
        def get_median(a):
            a = np.sort(list(set(a)))
            if len(a) % 2 == 0:
                return a[len(a) // 2 - 1]
            else:
                return a[len(a) // 2]

        median_value = get_median(time_series)
        index = [i for i in range(len(time_series)) if
                 i == 0 or i == len(time_series) - 1 or time_series[i] == median_value]
        return index

    def scale_time_series(index,time_series,scale_factor):
        # Step 2: select separator observations from attention observations with scale_factor
        separator = [index[i * scale_factor] for i in range(0, len(index) // scale_factor + 1) if
                     i * scale_factor < len(index)]
        if separator[-1] < index[-1]:
            separator.append(len(time_series) - 1)

        # Step 3: segment the time series into sub-series by the location of separator observations
        sub_series = [time_series[separator[i] + 1:separator[i + 1] + 1]  for i in
                      range(len(separator) - 1) if separator[i+1] -separator[i]>0]

        # Step 4: calculate the means of the sub-series segmented by separator observations as scaled_time_series
        scaled_time_series = [np.mean(i) for i in sub_series]

        return scaled_time_series

    # Step 1 find index of attention observations
    index=[]
    if scaling_model=="MS":
        index=index_t_th(time_series)
    elif scaling_model=="PAS":
        index=index_peak_points(time_series)
    elif scaling_model=="OAS":
        index=index_most_occurrences(time_series)
    elif scaling_model=="MAS":
        index=index_median(time_series)

    # step2 - step 4
    scaled_time_series=scale_time_series(index,time_series,scale_factor)
    return scaled_time_series

def shannon_entropy(time_series):
    """Return the Shannon Entropy of the sample data.

    Args:
        time_series: Vector or string of the sample data

    Returns:
        The Shannon Entropy as float value
    """
    occurence_count = Counter(time_series)
    freq_list=np.array(list(occurence_count.values()))
    ent=-np.sum([freq * (np.log2(freq)-np.log2(np.sum(freq_list)))/np.sum(freq_list) for freq in freq_list])
    return ent


# #example
# time_series=[1,2,3,4,5,3,1,3,0,4,2,1,3]
# scale_factor=2 #int
# scaling_model="PAS" # one of ["MS", "PAS", "OAS", "MAS"]
#
# time_series_scaled=attention_scale(time_series, scale_factor, scaling_model)
#
# #calculate Shonnon entropy
# e=shannon_entropy(time_series_scaled)
# print(e)