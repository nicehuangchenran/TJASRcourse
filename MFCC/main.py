### This program implements the MFCC
### Implement by 黄晨冉 2050664 for ASR assigment1
import matplotlib.pyplot as plt
import numpy as np
import librosa

def plot_time(data, sample_rate):
    """
    :param data: the audio data
    :param sample_rate: sample rate(44100Hz here)
    :return: no return
    """
    time = np.arange(0, len(data)) * (1.0 / sample_rate)
    plt.figure(figsize=(20, 5))
    plt.plot(time, data)
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

def plot_spectrogram(spec, note):
    """
    :param spec: the spectrogram
    :param note: label for axis y
    :return: no return
    """
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Frames')
    plt.ylabel(note)
    plt.tight_layout()
    plt.show()

def pre_emphasis(audio, factor=0.97):
    """
    alpha: pre_emphasis factor
    """
    for i in range(1, int(len(audio)), 1):
        audio[i - 1] = audio[i] - factor * audio[i - 1]
    # draw diagrams
    plot_time(audio, sr)
    return audio

def divide_frame(audio, sample_rate):
    """
    divide the audio
    """
    sig_len = len(audio)  # the length of signal
    frame_len = int(sample_rate * 0.025)  # the length of frame(25 ms)
    frame_mov = int(sample_rate * 0.010)  # the frame movement(10 ms)
    frame_num = int(np.ceil((sig_len - frame_len) / frame_mov))  # frame number

    # pad zeros
    zero_num = (frame_num * frame_mov + frame_len) - sig_len
    zeros = np.zeros(zero_num)

    # concat audio with zeros
    filled_signal = np.concatenate((audio, zeros))
    # extract the frame time
    indices = np.tile(np.arange(0, frame_len), (frame_num, 1)) + \
              np.tile(np.arange(0, frame_num * frame_mov, frame_mov), (frame_len, 1)).T

    # get the audio
    indices = np.array(indices, dtype=np.int32)
    divided_audio = filled_signal[indices]
    return divided_audio, frame_len

def windowing(emp_audio):
    """
    hamming window for pre-emphasis audio
    """
    divide_audio, frame_length = divide_frame(emp_audio, sr)  # frame dividing
    hamming_win = np.hamming(frame_length)  # hamming window
    windowed = divide_audio * hamming_win  # windowing
    # draw diagram
    plot_time(windowed[188], sr)
    return windowed

def STFT(windowed_audio):
    """
    STFT-Short-Time Fourier Transform
    :return: the STFT result
    """
    n_fft = 512  # points of FFT
    magnitude = np.absolute(np.fft.rfft(windowed_audio, n_fft))  # the magnitude
    power_audio = (1.0 / n_fft) * (magnitude ** 2)  # the powered audio

    #draw power spectrum
    plt.figure(figsize=(10, 5))
    plt.imshow(power_audio.T, origin='lower', aspect='auto')
    plt.colorbar()
    plt.xlabel("Frames")
    plt.ylabel("Frequencies")
    plt.ylim(0)
    plt.title("Power Spectrum")
    plt.show()
    return power_audio

def mel_filter(stft_audio):
    """
    :return: the filtered data, energe
    """
    n_fft = 512  # as above
    low_mel = 300  # lowest mel-filter value
    high_mel = 1125 * np.log(1 + (sr / 2) / 700)  # (f = sr / 2(Nyquist theorem)) with the function in the slides
    n_filters = 26  # number of filters(26 - 40)
    points = np.linspace(low_mel, high_mel, n_filters + 2)  # generate sequential points(+2 for index convenience)
    inverses = 700 * (np.e ** (points / 1125) - 1)  # calculate the frequency using the inverse function like hi_mel

    filter_bank = np.zeros((n_filters, int(n_fft / 2 + 1)))  # the filter bank
    f = (n_fft / 2) * inverses / (sr / 2)

    # generate filters
    for i in range(1, n_filters + 1):
        left = int(f[i - 1])
        center = int(f[i])
        right = int(f[i + 1])
        # f(m-1)<k<f(m)
        for j in range(left, center):
            filter_bank[i - 1, j + 1] = (j + 1 - f[i - 1]) / (f[i] - f[i - 1])
        # f(m)<k<f(m+1)
        for j in range(center, right):
            filter_bank[i - 1, j + 1] = (f[i + 1] - (j + 1)) / (f[i + 1] - f[i])
    # apply the filters to the audio
    energy = np.sum(stft_audio, 1)  # get the energy
    filtered_data = np.dot(stft_audio, filter_bank.T)  # dot product

    # plot as a freq-spectrogram
    plot_spectrogram(filtered_data.T, 'Filter Banks')
    return filtered_data, energy

def log(filtered_audio):
    filtered_audio = np.where(filtered_audio == 0, np.finfo(float).eps,
                             filtered_audio)  # if zero then replace with eps, else no change
    filtered_audio = np.log10(filtered_audio)
    log_audio = 20 * filtered_audio  # log
    return log_audio

def DCT(logged_audio, n_mfcc=26, n_ceps=12):
    """
    :param logged_audio: log10(mel-filtered data)
    :param n_mfcc: number of MFCC
    :param n_ceps: number of cepstral coefficients
    :return: the DCTed audio(keep only 12 frames of 26)
    """
    transpose = logged_audio.T
    len_data = len(transpose)
    dct_audio = []
    for j in range(n_mfcc):
        temp = 0
        for m in range(len_data):
            temp += (transpose[m]) * np.cos(j * (m + 0.5) * np.pi / len_data)
        dct_audio.append(temp)
    mfcc = np.array(dct_audio[1:n_ceps + 1])
    plot_spectrogram(mfcc, "MFCC coefficients")
    return mfcc

def delta(dcted_audio, k=1):
    """
    :param lift_audio: liftered audio
    :param k: the time gap for first-rank derivation
    :return: the delta array
    """
    delta_audio = []
    transpose = dcted_audio.T
    q = len(transpose)  # the dimension of the mfcc
    for t in range(q):
        if t < k:
            delta_audio.append(transpose[t + 1] - transpose[t])
        elif t >= q - k:
            delta_audio.append(transpose[t] - transpose[t - 1])
        else:
            denominator = 2 * sum([i ** 2 for i in range(1, k + 1)])
            numerator = sum([i * (transpose[t + i] - transpose[t - i]) for i in range(1, k + 1)])
            delta_audio.append(numerator / denominator)
    return np.array(delta_audio)

def dynamic_fearture_extraction(dcted_audio):
    delta_audio= delta(dcted_audio)
    delta2_audio = delta(delta_audio)
    plot_spectrogram(delta2_audio, "Dynamic feartures")
    return delta2_audio

def normalization(dynamic_feature):
    mean = np.mean(dynamic_feature, axis=1)
    std = np.std(dynamic_feature, axis=1)
    transpose = dynamic_feature.T
    for i in range(len(transpose)):
        transpose[i] = (transpose[i] - mean) / std
    return transpose.T

if __name__ == '__main__':
    # read audio
    path="demo_2050664.wav"
    sr=8000 #sample rate
    audio,sample_rate = librosa.load(path, sr=8000)
    plot_time(audio, sr)
    # step 1 pre_emphasis
    pre_audio=pre_emphasis(audio)
    # step 2 windowing
    windowed_audio=windowing(pre_audio)
    #step 3 STFT
    power_audio=STFT(windowed_audio)
    #step 4 mel-filter
    filtered_audio,energe=mel_filter(power_audio)
    #step 5 log()
    logged_audio=log(filtered_audio)
    #step 6 DCT
    dcted_audio=DCT(logged_audio)
    #step 7 dynamic feature extraction
    dynamic_feature=dynamic_fearture_extraction(dcted_audio)
    # step 8 feature transformation
    feature_normalized=normalization(dynamic_feature)

















