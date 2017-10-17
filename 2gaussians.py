import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib.patches import Ellipse
import sklearn.datasets as ds
from bvestimator.bvestimator import BVEstimator


sigma = 0.5
cov = [[sigma, 0], [0, sigma]]
mu1 = 2.0
mu2 = 4.0
mu3 = 6.0
sz = 10
clf = DecisionTreeClassifier()


input_dims = 2
def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).
    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


def pdf(x, mu, sigma):
    return (1/(sigma * np.sqrt(2 * np.pi))) * np.exp( -(x - mu)**2 / (2 * sigma**2) )


#plot histogram of a distribution without showing it, call plt.show() to do so
def graph(dist, pdf, color):
    plt.plot(dist, pdf, linewidth=2, color=color)

def interpolate_to_get_Px(D1, D2, Px_d1_c1, Px_d2_c2, x, py):
    return np.interp(x, D1, Px_d1_c1) * py + np.interp(x, D2, Px_d2_c2) * py

def compute_bayes(pxc, py, px):
	return pxc*py/px

def combine2ds(Xy1, Xy2):
    X1, y1 = Xy1
    X2, y2 = Xy2
    X=np.row_stack((X1, X2))
    y=np.hstack((y1,y2))
    return X, y

def main():
    '''

   X, y = ds.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0,
                        n_repeated=0, n_classes=2, n_clusters_per_class=2,
                        weights=None, flip_y=0.01, class_sep=1.0, hypercube=False, shift=0.21,
                        scale=1.0, shuffle=True,random_state=None)
    '''
    #X, y = ds.make_circles()
    #X, y = ds.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0,
    #                              n_repeated=0, n_classes=2, n_clusters_per_class=2,
    #                              weights=None, flip_y=0.01, class_sep=2.0, hypercube=True, shift=0.31,
    #                              scale=1.0, shuffle=True, random_state=None)
    #X, y = ds.make_hastie_10_2()
    #X, y = combine2ds(ds.make_circles(), ds.make_moons())
    #   X, y = ds.make_checkerboard((3, 3), 2)
    #X2, y2 = ds.make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=0.33)
    #X3, y3, = ds.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0,
    #                                                     n_repeated=0, n_classes=2, n_clusters_per_class=2,
    #                                                     weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.21,
    #                                                     scale=1.0, shuffle=True, random_state=None)
    #X=np.row_stack((X1, X2, X3))
    #y = np.hstack((y1,y2,y3))
    #X, y = ds.load_breast_cancer(True)\
    #X, y = ds.make_blobs(n_samples=10, n_features=2, centers=2, cluster_std=0.33)
    X, y = ds.make_hastie_10_2()
    for i, item in enumerate(y):
        if item < 0:
            y[i]=0
        else:
            y[i]=1
    y = y.astype(int)
    est = BVEstimator()
    est.estimate(X, y, LogisticRegression())
    #https://pdfs.semanticscholar.org/48e0/27a29968eb27e31029999f187eb63510255f.pdf
    #MultiBoosting: A Technique for Combining Boosting and Wagging
    #http://ai.stanford.edu/~ronnyk/biasVar.pdf

if __name__ == '__main__':
    main()

