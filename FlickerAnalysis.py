import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft,fftfreq
import pdb
fs = 10000
import struct

def read_16bit_signed_binary(file_path):
    """
    Reads a binary file containing 16-bit signed integers.

    Args:
        file_path (str): The path to the binary file.

    Returns:
        list: A list of 16-bit signed integers read from the file.
    """
    data = []
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(2)  # Read 2 bytes (16 bits) at a time
            if not chunk:
                break  # End of file
            value = struct.unpack('<h', chunk)[0]  # Unpack as little-endian signed short
            data.append(value)
    return data

# Example usage:
file_path = 'F1.bin'  # Replace with your file path
data = read_16bit_signed_binary(file_path)

#print(data)
#df = pd.read_csv('SSVEP.csv')
#plt.plot(df['Time (s)'],df['Channel 1 (V)'])
#plt.show()
plt.plot(data)
plt.show()
srange1 = (52000,153000)#F1#(37000,135700)#E2#(47000,147000)#D1#(45000,144000)#C1#(28000,128000)#A2#(147000,346800)#A1#(44000,144000)#H4#(34000,135000)#H2#(45280,146000)#H1
#srange1 = (1650000,1760000)#(30770,35822) #Video 1: (134000,233800) Video 2:(344756,444022) Video 3: (563682,663680) Video 4: (780000,900000) Video 5: (1000000,1120000) Video 6: (1225010,1324300) Video 7: (1425000,1540000) Video 8: (1650000,1760000)
Flicker = data#df['Channel 1 (V)'].values
Flick1 = Flicker[srange1[0]:srange1[1]]
plt.plot(Flick1)
plt.show()
N = len(Flick1)
T = 1./fs
YF = fft(Flick1)
XF = fftfreq(N, T)[:N//2]
plt.plot(XF, 2.0/N * np.abs(YF[0:N//2]))
plt.show()