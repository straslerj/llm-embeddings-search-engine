import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
Something in how I read my DataFrame gets tricky (specifically due to poor naming that is going to be a lot of work to get around) makes the plots wrong, 
so I figured it would be easier to simply plug in the data to a quick script since there is relatively few data points to manually enter. Please ignore
the poor coding standards of this script... it was quick and dirty. The data is pulled from results/<mode>_results_STATS.csv
"""

# NFL data

means = [
    0.4301993762484834,
    0.31182425765650745,
    0.333954009192088,
    0.2902715327100286,
    0.5697966280711603,
    0.4744308906949208,
]
std_devs = [
    0.3180499251083353,
    0.27828509747004326,
    0.2705386681673722,
    0.2529973981304602,
    0.2769086110888851,
    0.318439841363638,
]

llm_labels = ["Context", "GPT-2", "GPT-2 Large", "DistilGPT2", "Phi-2", "Phi-3 Mini"]

sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.alpha": 0.7})
plt.figure(figsize=(15, 6))
sns.barplot(x=llm_labels, y=means, hue=llm_labels, palette="colorblind")
plt.errorbar(
    x=llm_labels, y=means, yerr=std_devs, fmt="none", ecolor="black", capsize=5
)
plt.title("Mean and Standard Deviation by LLM (NFL Rulebook)")
plt.xlabel("LLM")
plt.ylabel("Similarity Score")
plt.ylim(0, 1.0)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.tight_layout()

plt.savefig("results/figures/nfl.png", dpi=300)

# Syllabus data

means = [
    0.41155825803677243,
    0.40195179261888064,
    0.3451283392806848,
    0.3994512071212133,
    0.5432606716950734,
    0.5735278179248174,
]
std_devs = [
    0.22243069952505273,
    0.232357247275062,
    0.19116895630275693,
    0.22843016587528148,
    0.19562349715841557,
    0.2251201426695241,
]

llm_labels = ["Context", "GPT-2", "GPT-2 Large", "DistilGPT2", "Phi-2", "Phi-3 Mini"]

sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.alpha": 0.7})
plt.figure(figsize=(15, 6))
sns.barplot(x=llm_labels, y=means, hue=llm_labels, palette="colorblind")
plt.errorbar(
    x=llm_labels, y=means, yerr=std_devs, fmt="none", ecolor="black", capsize=5
)
plt.title("Mean and Standard Deviation by LLM (Syllabus)")
plt.xlabel("LLM")
plt.ylabel("Similarity Score")
plt.ylim(0, 1.0)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.tight_layout()

plt.savefig("results/figures/syllabus.png", dpi=300)
