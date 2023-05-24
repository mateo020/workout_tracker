import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from website.DataTransformation import LowPassFilter, PrincipalComponentAnalysis
import numpy as np
import joblib
from sklearn.cluster import KMeans

# acc_df= pd.read_csv("../website/Fit_2023-05-20T11.55.56.578_C8AD2DA60669_Accelerometer.csv")

# gyr_df= pd.read_csv("../website/Fit_2023-05-20T11.55.56.578_C8AD2DA60669_Gyroscope.csv")



def make_dataset(csv_file1, csv_file2):
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()
    
    acc_df = pd.read_csv(csv_file1)
    gyr_df = pd.read_csv(csv_file2)
    
        
    acc_df.index = pd.to_datetime(acc_df['epoc (ms)'], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df['epoc (ms)'], unit="ms")

    del acc_df['epoc (ms)']
    del acc_df['timestamp (-0400)']
    del acc_df['elapsed (s)']


    del gyr_df['epoc (ms)']
    del gyr_df['timestamp (-0400)']
    del gyr_df['elapsed (s)']
    
    

    data_merged = pd.concat([acc_df.iloc[:,:3], gyr_df], axis=1)
    
    data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gye_x",
    "gyr_y",
    "gyr_z"]
    
    
# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------
    sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gye_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean"}
    
    data_merged[:1000].resample(rule="200ms").mean().apply(sampling)
    
    days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
    data_resampled = pd.concat([df.resample(rule = "200ms").apply(sampling).dropna() for df in days])
    data_resampled.info()
    



    return data_resampled









# This function aggregates a list of values using the specified aggregation
# function (which can be 'mean', 'max', 'min', 'median', 'std')
def aggregate_value( aggregation_function):
    # Compute the values and return the result.
    if aggregation_function == "mean":
        return np.mean
    elif aggregation_function == "max":
        return np.max
    elif aggregation_function == "min":
        return np.min
    elif aggregation_function == "median":
        return np.median
    elif aggregation_function == "std":
        return np.std
    else:
        return np.nan

# Abstract numerical columns specified given a window size (i.e. the number of time points from
# the past considered) and an aggregation function.
def abstract_numerical( data_table, cols, window_size, aggregation_function):

    # Create new columns for the temporal data, pass over the dataset and compute values
    for col in cols:
        data_table[
            col + "_temp_" + aggregation_function + "_ws_" + str(window_size)
        ] = (
            data_table[col]
            .rolling(window_size)
            .apply(aggregate_value(aggregation_function))
        )

    return data_table






# Find the amplitudes of the different frequencies using a fast fourier transformation. Here,
# the sampling rate expresses the number of samples per second (i.e. Frequency is Hertz of the dataset).
def find_fft_transformation( data, sampling_rate):
    # Create the transformation, this includes the amplitudes of both the real
    # and imaginary part.
    transformation = np.fft.rfft(data, len(data))
    return transformation.real, transformation.imag

# Get frequencies over a certain window.
def abstract_frequency( data_table, cols, window_size, sampling_rate):

    # Create new columns for the frequency data.
    freqs = np.round((np.fft.rfftfreq(int(window_size)) * sampling_rate), 3)

    for col in cols:
        data_table[col + "_max_freq"] = np.nan
        data_table[col + "_freq_weighted"] = np.nan
        data_table[col + "_pse"] = np.nan
        for freq in freqs:
            data_table[
                col + "_freq_" + str(freq) + "_Hz_ws_" + str(window_size)
            ] = np.nan

    # Pass over the dataset (we cannot compute it when we do not have enough history)
    # and compute the values.
    for i in range(window_size, len(data_table.index)):
        for col in cols:
            real_ampl, imag_ampl = find_fft_transformation(
                data_table[col].iloc[
                    i - window_size : min(i + 1, len(data_table.index))
                ],
                sampling_rate,
            )
            # We only look at the real part in this implementation.
            for j in range(0, len(freqs)):
                data_table.loc[
                    i, col + "_freq_" + str(freqs[j]) + "_Hz_ws_" + str(window_size)
                ] = real_ampl[j]
            # And select the dominant frequency. We only consider the positive frequencies for now.

            data_table.loc[i, col + "_max_freq"] = freqs[
                np.argmax(real_ampl[0 : len(real_ampl)])
            ]
            data_table.loc[i, col + "_freq_weighted"] = float(
                np.sum(freqs * real_ampl)
            ) / np.sum(real_ampl)
            PSD = np.divide(np.square(real_ampl), float(len(real_ampl)))
            PSD_pdf = np.divide(PSD, np.sum(PSD))
            data_table.loc[i, col + "_pse"] = -np.sum(np.log(PSD_pdf) * PSD_pdf)

    return data_table







def filter_out_noice(df):
    
    df_lowpass = df.copy()
    lowPass = LowPassFilter()

    fs = 1000/200
    cutoff = 1.2

    df_lowpass = lowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order = 5)

    predictor_columns = list(df.columns)

    for col in predictor_columns:
        df_lowpass = lowPass.low_pass_filter(df_lowpass,col,fs,cutoff,order=5)
        df_lowpass[col] = df_lowpass[col + "_lowpass"]
        del df_lowpass[col+"_lowpass"]

    df_pca = df_lowpass.copy()
    
    PCA = PrincipalComponentAnalysis()

    pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

    plt.figure(figsize=(10,10))
    plt.plot(range(1,len(predictor_columns) +1), pc_values)
    plt.xlabel("principal component number")
    plt.ylabel("explained varaince ")
    plt.show()

    df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)
    
    df_squared = df_pca.copy()
    acc_r = df_squared["acc_x"]**2 + df_squared["acc_y"]**2 + df_squared["acc_z"]**2
    gyr_r = df_squared["gye_x"]**2 + df_squared["gyr_y"]**2 + df_squared["gyr_z"]**2


    df_squared["acc_r"] = np.sqrt(acc_r)
    df_squared["gyr_r"] = np.sqrt(gyr_r)
    
    # --------------------------------------------------------------
    # Temporal abstraction
    # --------------------------------------------------------------


    df_temporal = df_squared.copy()
    

    predictor_columns= predictor_columns + ["acc_r", "gyr_r" ]

    ws = int(1000 / 200)
    for col in predictor_columns:
        df_temporal = abstract_numerical(df_temporal,[col], ws, "mean")
        df_temporal = abstract_numerical(df_temporal,[col], ws, "std")
        
    # --------------------------------------------------------------
    # Frequency features
    # --------------------------------------------------------------

    df_freq = df_temporal.copy().reset_index()
   

    fs = int(1000/200)
    ws = int(2800 / 200)

    df_freq = abstract_frequency(df_freq,predictor_columns,ws,fs)

        
    df_freq= df_freq.dropna()
    df_freq.iloc[::2]
    
    
    df_cluster = df_freq.copy()
    cluster_columns  = ["acc_y", "acc_y", "acc_z"]
    k_values = range(2,10)
    intertias = []

    for k in k_values:
        subset = df_cluster[cluster_columns]
        kmeans = KMeans(n_clusters=k, n_init=2, random_state=0)
        cluster_labels = kmeans.fit_predict(subset)
        intertias.append(kmeans.inertia_)
        

    plt.figure(figsize=(10,10))
    plt.plot(k_values, intertias)
    plt.xlabel("k")
    plt.ylabel("sum of squared distances")
    plt.show()

    kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
    subset = df_cluster[cluster_columns]
    df_cluster["cluster"] = kmeans.fit_predict(subset)
    
    fig = plt.figure(figsize = (15,15))
    ax = fig.add_subplot(projection="3d")
    for c in df_cluster["cluster"].unique():
        subset = df_cluster[df_cluster["cluster"] == c]
        ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    plt.legend()
    plt.show()
    
    
    
    basic_features = ["acc_x", "acc_y", "acc_z", "gye_x", "gyr_y", "gyr_z", ]
    square_features = ["acc_r","gyr_r"]
    pca_features = ["pca_1", "pca_2", "pca_3"]
    times_features = [f for f in df_cluster.columns if "_temp_" in f]
    freq_featues = [f for f in df_cluster.columns if (("_freq" in f) or ("_pse" in f))]
    cluster_featues = ["cluster"]

    print("basic featues:", len(basic_features))
    print("square featues:", len(square_features))
    print("PCA features: ", len(pca_features))
    print("Times featues:", len(times_features))
    print("freqeuncy featues:", len(freq_featues))
    print("cluster features:", len(cluster_featues))
        
    features_set_1 = list(set(basic_features))
    features_set_2 =list(set(basic_features + square_features + pca_features))
    features_set_3 = list(set(features_set_2 + times_features))
    features_set_4 = list(set(features_set_3 + freq_featues + cluster_featues))
        
    X = df_cluster.loc[:, df_cluster.columns != 'epoc (ms)']
    model, ref_cols, target = joblib.load("/Users/mateo/OneDrive/Desktop/tracking-barbell-exercises/models/model.pkl")
    predictions = model.predict(df_cluster[ref_cols])
    
    return predictions



def predicted_exercises(predictions):
    
    predictions = np.unique(predictions)
    exercise_list = []
    current_exercise = None

    for prediction in predictions:
        if prediction != 'rest':
            if current_exercise is None or current_exercise != prediction:
                exercise_list.append(prediction)
                current_exercise = prediction
        else:
            current_exercise = None

    return exercise_list
    
        
        