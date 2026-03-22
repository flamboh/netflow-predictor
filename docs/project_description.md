# Primary questions and goals

- Can the MAAD Structure and Spectrum analyses inform a machine learning model of the next time window's unique Source or Destination IP Address count?
  - At what specificity can we confidently predict this next window?
    - Direction of change from previous window.
    - Difference from previous window.
    - Exact count of unique IP Address in next window.
  - What model architectures work well?
    - LSTM/RNN
    - Transformer
    - S4/timeseries foundational models
  - We can test this through use of two separate feature sets:
    1. excluding the multifractal analyses
    2. including the multifractal analyses
  - What features are even useful?
    - Start with Packet/Byte/Flow totals and counts across TCP, UDP, ICMP
    - Mean, median, and standard deviation of flow size in bytes.
    - Unique IP Counts (source/destination) and the associated mean, median, and standard deviation
    - Unique IP pair counts
      - Will have to build pipeline to calculate this.
    - Since we're working with a time series, we will also include the values for these features of recent past values (AKA their lags)
      - Will start with last 12 time windows, could refine this later
    - Could later include unique counts of different prefix resolutions (e.g. how many source /16 prefixes are present)
      - I don't have this data right now, so it could be saved for future exploration
    - How do we best encode Spectrum/Structure?
      - Initially, use all points representing their graphs.
  - This process requires a few steps:
    1. Curating a dataset from UO NetFlow data
       - Partitioning into training, validation, and testing subsets
       - How to partition initially
         - Take 1 term, 70% for training, 20% for validation, 10% for testing
         - Iterate on this as necessary
    2. Training the above mentioned models
       - We'll save exploration of hyper-parameters for later.
    3. Evaluation and iteration on performance
       - Selecting different address structure encodings
       - Testing on other datasets (UGR'16, etc.)

The initial goal here is to 'quickly' test several models to verify the efficacy of the MAAD analyses for informing some metric of network traffic. Unique IP Address count was selected due to its semantic relation to the MAAD analyses, they are built from the set of unique IP Addresses.

To accomplish the training of these models, I'll be using PyTorch, the data I've gathered in the ../netflow-analysis project, and the existing data.
