import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import mode
import matplotlib.pyplot as plt
import matplotlib
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.patches as mpatches
from sklearn.feature_selection.univariate_selection import SelectKBest
from sklearn import preprocessing

class BVEstimator:
    def __init__(self):
        self.skf = KFold(n_splits=3, shuffle=True)

    def estimate(self, X, y, classifier):
        clf_name = str(classifier.__class__).split('.')[-1]
        true_values = np.zeros(y.shape)
        trials = []
        all_true_values = []
        X = SelectKBest(k=2).fit_transform(X, y)
        scaler = preprocessing.MinMaxScaler()
        X = scaler.fit_transform(X, y)
        h=.02
        minx = np.min((np.min(X[0]), np.min(X[1]))) - 1.0
        maxx = np.max((np.max(X[0]), np.max(X[1]))) + 1.0
        xx, yy = np.mgrid[minx:maxx:h, minx:maxx:h]
        N = 10
        grid = np.c_[xx.ravel(), yy.ravel()]

        for k in range(N):
            predictions = np.zeros(y.shape)
            for train_index, test_index in self.skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                true_values[test_index] = y_test
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                predictions[test_index] = y_pred

            trials.append(predictions)
            all_true_values.append(true_values)

        full_table = np.array(trials)
        full_true = np.array(all_true_values)
        central_tendency = np.zeros(y.shape)
        error = np.zeros(len(y))
        bias = np.zeros(len(y))
        variance = np.zeros(len(y))
        T = np.zeros(len(set(y)))
        for i, t in enumerate(T):
            T[i] = len(np.extract(y == t, y))/len(y)

        Z = classifier.predict_proba(grid)[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)

        cm = matplotlib.colors.ListedColormap(['red', 'blue'])
        ax = plt.subplot2grid((1, 3), (0, 0))
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.3)


        ax.scatter(X[:, 0], X[:, 1], c=y,
                   cmap="RdBu",
                   edgecolor="white", alpha=1.0,linewidth=1)
        ax.set_title(clf_name)
        ax.set(aspect="equal",
               xlabel="$X_1$", ylabel="$X_2$")

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        Xx = []
        Yy = []
        bv = []



        for i in range(len(y)):
            md, ct = mode(full_table[:, i])
            central_tendency[i] = md
            err_val = []
            for k in range(10):
                if full_table[k, i] != full_true[k, i]:
                    error[i] += 0.1
                    err_val.append(full_table[k, i])

            if error[i] > 0:
                wi = err_val.count(central_tendency[i]) * 0.1
                U = (wi - T[y[i]])
                variance[i] = wi * (1 - wi) / (N - 1)
                bias[i] = U**2 - variance[i]

                print('bias %f variance %f error %f' % (bias[i], variance[i], error[i]))
                if (bias[i] > 0 or variance[i] > 0):
                    Xx.append(X[i, 0])
                    Yy.append(X[i, 1])
                    if bias[i] > variance[i]:
                        bv.append(0)
                    else:#variance error more common and difficult favor it
                        bv.append(1)
            #    ax.annotate(str(bias[i]) + ',' + str(variance[i]), (X[i, 0], X[i, 1]))

        ax = plt.subplot2grid((1, 3), (0, 1))

        ax.contourf(xx, yy, Z, cmap="RdBu", alpha=.3)
        bv_colors = ['green', 'yellow']
        cm = matplotlib.colors.ListedColormap(bv_colors)
        ax.scatter(Xx, Yy, c=bv,cmap=cm,
                   edgecolor="white", alpha=1.0,linewidth=1)
        recs = []
        for i in range(0, len(bv_colors)):
            recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=bv_colors[i]))
        ax.legend(recs, ['bias', 'variance'], loc=4)
        ax.set(aspect="equal",
               xlabel="$X_1$", ylabel="$X_2$")

        ax.set_title('Error causes')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())


        if (len(Xx) > 0):
            gp = GaussianProcessRegressor()
            Xx = np.column_stack((np.asarray(Xx), np.asarray(Yy)))
            gp.fit(Xx, bv)
            Z = gp.predict(grid)

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)

            ax = plt.subplot2grid((1, 3), (0, 2))
            #
            ax.contourf(xx, yy, Z, label=['bias', 'variance'], cmap=cm, alpha=.3)
            ax.set(aspect="equal",
                   xlabel="$X_1$", ylabel="$X_2$")
            ax.legend()
            ax.set_title('Error regions')
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())



        plt.show()

