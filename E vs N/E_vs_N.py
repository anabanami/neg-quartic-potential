from pathlib import Path
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 200

data_file = Path("data.json")

data = json.loads(data_file.read_text()) 

N_list = data["N"]
E_lists = data["E"]


def Extract(lst, n):
    E_n = []
    for item in lst:
        try:
            E_n.append(item[n])
        except IndexError:
            pass
        continue
    return E_n


for n in range(5):
    E_n = Extract(E_lists, n)
    evalue = E_n[-1]
    if n <= 1:
        plt.plot(N_list, E_n, label=fR"$E_{n} = {evalue:.06f}$")

    else:
        plt.plot(N_list[n-1:], E_n, label=fR"$E_{n} = {evalue:.06f}$")

# plt.title("Eigenvalue convergence number of basis states (N)")
plt.legend()
plt.ylabel("Energy")
plt.xlabel("N")
plt.show()








