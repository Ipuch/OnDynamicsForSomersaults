import pickle
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

filename = "miller_exp.bo"
file = open(f"{filename}", "rb")
data = pickle.load(file)
q1 = np.hstack((data[0][0]["q"], data[0][1]["q"]))

filename = "miller_exp_2.bo"
file = open(f"{filename}", "rb")
data = pickle.load(file)
q2 = np.hstack((data[0][0]["q"], data[0][1]["q"]))

fig = make_subplots(rows=5, cols=3)
# for each dof display q trajectories on each subplot
for i in range(15):
    # plot in red
    fig.add_trace(
        go.Scatter(x=np.arange(0, q1.shape[1]), y=q1[i, :], name="q1", mode="lines", line=dict(color="red")),
        row=i // 3 + 1,
        col=i % 3 + 1,
    )
    # plot in blue
    fig.add_trace(
        go.Scatter(x=np.arange(0, q2.shape[1]), y=q2[i, :], name="q2", mode="lines", line=dict(color="blue")),
        row=i // 3 + 1,
        col=i % 3 + 1,
    )
fig.show()
