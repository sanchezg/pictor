import matplotlib.pyplot as plt


def plot_histogram(x_label, y_label, data, save=False, aname=None,
                   normed=False, step=100):
    """Plots an histogram with the array data passed by parameter."""
    plt.clf()
    n, bins, p = plt.hist(data, bins=100, normed=normed)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    path = '../out/'
    if save:
        if aname is not None:
            filename = 'hist_' + x_label + aname + '.png'
        else:
            filename = 'hist_' + x_label + '.png'
        plt.savefig(path + filename, bbox_inches='tight')
    else:
        plt.show()
    return zip(bins, n)


def plot_bar_chart(data, feature, labels=None):
    """Plots a bar chart with the data argument."""
    xpos = np.arange(len(data))
    categories = []
    category_count = []
    for category, count in data:
        categories.append(category)
        category_count.append(count)

    plt.bar(xpos, category_count, .35, color='y', align='center')
    plt.ylabel('Categories count')
    plt.xticks(xpos, categories)
    plt.title(feature + ' values count')

    plt.show()
    return


def plot_scatter(xlabel, ylabel, xfeat, yfeat, save=False, aname=None):
    """."""
    plt.clf()
    plt.scatter(xfeat, yfeat)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    path = '../out/'
    if save:
        if aname is not None:
            filename = 'scatter_' + xlabel + '_x_' + ylabel + aname + '.png'
        else:
            filename = 'scatter_' + xlabel + '_x_' + ylabel + '.png'
        plt.savefig(path + filename, dpi=1000, bbox_inches='tight')
    else:
        plt.show()
    return