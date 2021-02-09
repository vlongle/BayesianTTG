import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np

plt.style.use('ggplot')
rcParams['figure.figsize'] = 10, 5
rcParams["legend.loc"] = 'lower right'

n = 100
xs = np.linspace(0, 2, n)

def plot_linear():
    f1 = lambda x: x
    f2 = lambda x: x * 0.5
    l1 = '$f_1(x)=x$'
    l2 = "$f_2(x) = (1/2)x$"
    title = "Linear division rule"
    save = "./lin_div.png"
    plot_div(f1, f2, l1, l2, title,save)

def plot_quadratic():
    f1 = lambda x: x**2
    f2 = lambda x: (x/2)**2
    l1 = '$f_1(x)=x^2$'
    l2 = "$f_2(x) = (1/4)x^2$"
    title = "Quadratic division rule"
    save = "./quad_div.png"
    plot_div(f1, f2, l1, l2, title,save)

def plot_exp():
    f1 = lambda x: np.exp(np.log(2) * x) - 1
    f2 = lambda x: np.exp(np.log(2) * (x/2)) - 1
    l1 = '$f_1(x)=e^{\log(2)x}-1$'
    l2 = "$f_2(x) = e^{(\log(2)/2)x}-1$"
    title = "Exponential division rule"
    save = "./exp_div.png"
    plot_div(f1, f2, l1, l2, title,save)


def plot_div(f1, f2, l1, l2, title, save):
    y1s = [f1(x) for x in xs]
    y2s = [f2(x) for x in xs]
    x = [0.5, 0.5]
    y = [f1(0.5), f2(0.5)]
    x2 = [1, 2]
    y2 = [f1(1), f2(2)]

    #plt.vlines(x, 0, y, linestyle="dashed", label="target")
    #plt.hlines(y, 0, x, linestyle="dashed")

    plt.vlines(x2, 0, y2, linestyle="dotted",  label="boundary")
    plt.hlines(y2, 0, x2, linestyle="dotted")

    #plt.scatter(x, y, color='black')
    plt.scatter(x2, y2, color='black')


    plt.plot(xs, y1s, label=l1)
    plt.plot(xs, y2s, label=l2)

    plt.legend(prop={'size': 15})
    plt.title(title)
    plt.xlabel("$x$")
    plt.ylabel("$f_{\\alpha}(x)$")
    plt.savefig(save)

plot_linear()
#plot_quadratic()
#plot_exp()