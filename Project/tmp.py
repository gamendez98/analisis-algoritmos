import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# %%

run_time = pd.read_csv("Project/run_times2.csv")
# %%
sns.regplot(
    data=run_time, x="n", y="run_time",
    marker="x", color=".3", line_kws=dict(color="r")
)
plt.show()

# %%

run_time['n2'] = run_time['n'] ** 2
sns.regplot(
    data=run_time, x="n2", y="run_time", order=1,
    marker="x", color=".3", line_kws=dict(color="r")
)
plt.show()

# %%

run_time_watts = pd.read_csv("Project/run_times_watts2.csv")
# %%
sns.regplot(
    data=run_time_watts, x="n", y="run_time",
    marker="x", color=".3", line_kws=dict(color="r")
)
plt.show()

# %%

run_time_watts['n2'] = run_time_watts['n'] ** 2
sns.regplot(
    data=run_time_watts, x="n2", y="run_time", order=1,
    marker="x", color=".3", line_kws=dict(color="r")
)
plt.show()

# %%

run_time_iter = pd.read_csv("Project/run_times_max_iter2.csv")
sns.scatterplot(
    data=run_time_iter, x="max_iters", y="run_time"
)
plt.show()

#%%



# %%

run_times_needed_iter = pd.read_csv("Project/run_times_needed_iter2.csv")
sns.scatterplot(data=run_times_needed_iter, x="N", y="needed_iter")
# Calculate and plot the moving average
window_size = 50
run_times_needed_iter['moving_avg'] = (
    run_times_needed_iter['needed_iter']
    .rolling(window=window_size, center=True)
    .mean()
)
#sns.lineplot(data=run_times_needed_iter, x="N", y="moving_avg", color="red")
plt.show()
