import math
import numpy as np
import matplotlib.pyplot as plt
import random
from fichiers_python_sec5 import hilbertcurve
from fichiers_python_sec5 import chain_to_image_functions as ch

def gaussienne(x, m, sig):
    return 1/(((2*np.pi)**(1/2))*sig) * math.exp(-((x - m)**2/(2*(sig**2))))

def  classif_gauss2(Y,cl1,cl2,m1,sig1,m2,sig2):
    S = np.zeros(len(Y))
    for i in range(len(Y)):
        y = Y[i]
        if gaussienne(y, m1, sig1) >= gaussienne(y, m2, sig2):
            S[i] = cl1
        else:
            S[i] = cl2
    return S

def erreur_moyenne_gauss(T, X, cl1, cl2, m1, sig1, m2, sig2):
    erreurs = np.zeros(T)
    for i in range(T):
        X_bruite = bruit_gauss2(X, cl1, cl2, m1, sig1, m2, sig2)
        S = classif_gauss2(X_bruite,cl1,cl2,m1,sig1,m2,sig2)
        erreurs[i] = taux_erreur(X, S)
    return erreurs

def taux_erreur(A, B):
    if len(A) != len(B):
        return "probleme de taille"
    else:
        nb_diff = 0
        for i in range(len(A)):
            if (A[i] != B[i]):
                nb_diff += 1
        taux_err = nb_diff/len(A)
        return taux_err

def tirage_classe2(p1,p2,cl1,cl2):
    u = random.random()
    return (u <= p1)*cl1 + (1-(u <= p1))*cl2

def gauss2(Y, n, m1, sig1, m2, sig2):
    Mat_f = []
    for i in range(n):
        y = Y[i]
        bruit = [gaussienne(y, m1, sig1), gaussienne(y, m2, sig2)]
        Mat_f.append(bruit)
    return Mat_f

def forward2(Mat_f,n,A,p10,p20):
    alfa = []
    u = random.random() #utilise pour savoir si etat initial en 0 ou 1
    alfa1 = np.zeros(2)
    alfa1[0] = Mat_f[0][0]* ((u<=p10)*A[0][0] + (1-(u<=p10))*A[1][0])
    alfa1[1] = Mat_f[0][1]* ((u<=p10)*A[0][1] + (1-(u<=p10))*A[1][1])
    alfa.append(alfa1)

    for i in range(1, n):
        alfa_recursive = np.zeros(2)
        for j in range(2):
            sum = alfa[i-1][0]*A[0][j] + alfa[i-1][1]*A[1][j]
            alfa_recursive[j] = sum * Mat_f[i][j]
        alfa.append(alfa_recursive)
    return alfa

def scaled_forward2(Mat_f,t,A,p10,p20):
    alfa = []
    u = random.random() #utilise pour savoir si etat initial en 0 ou 1
    alfa1 = np.zeros(2)
    alfa1[0] = Mat_f[0][0]*p10
    alfa1[1] = Mat_f[0][1]*p20
    denominateur = alfa1[0]+alfa1[1]
    alfa1[0] = alfa1[0]/denominateur
    alfa1[1] = alfa1[1]/denominateur
    alfa.append(alfa1)

    for n in range(1, t):
        alfa_recursive = np.zeros(2)
        for i in range(2):
            sum = alfa[n-1][0]*A[0][i] + alfa[n-1][1]*A[1][i]
            sum = sum*Mat_f[n][i]
            alfa_recursive[i] = sum
        denominateur_rec = alfa_recursive[0] + alfa_recursive[1]
        alfa_recursive[0] = alfa_recursive[0]/denominateur_rec
        alfa_recursive[1] = alfa_recursive[1]/denominateur_rec
        alfa.append(alfa_recursive)
    return alfa

def backward2(Mat_f,n,A):
    beta = []
    for i in range(n-1):
        beta.append([0, 0])
    beta.append([1, 1])

    for t in range(n-2, -1, -1):
        beta_recursive = np.zeros(2)
        for i in range(2):
            sum = A[i][0]*Mat_f[t+1][0]*beta[t+1][0] + A[i][1]*Mat_f[t+1][1]*beta[t+1][1]
            beta_recursive[i] = sum
        beta[t] = beta_recursive
    return beta

def scaled_backward2(Mat_f,n,A):
    beta = []
    for i in range(n-1):
        beta.append([0, 0])
    beta.append([0.5, 0.5])

    for t in range(n-2, -1, -1):
        beta_recursive = np.zeros(2)
        for i in range(2):
            sum = A[i][0]*Mat_f[t+1][0]*beta[t+1][0] + A[i][1]*Mat_f[t+1][1]*beta[t+1][1]
            beta_recursive[i] = sum
        denominateur = beta_recursive[0] + beta_recursive[1]
        beta_recursive[0] /= denominateur
        beta_recursive[1] /= denominateur
        beta[t] = beta_recursive
    return beta

def Baum_Welch(Mat_f,n,cl1,cl2,A,p10,p20):
    S = np.zeros(n)
    alfa = scaled_forward2(Mat_f,n,A,p10,p20)
    beta = scaled_backward2(Mat_f, n, A)
    for t in range(n-1):
        sigma = [[0, 0], [0, 0]]

        denominateur = 0
        for k in range(2):
            for l in range(2):
                denominateur += alfa[t][k]*A[k][l]*Mat_f[t+1][l]*beta[t+1][l]

        for i in range(2):
            for j in range(2):
                sigma[i][j] = alfa[t][i]*A[i][j]*Mat_f[t+1][j]*beta[t+1][j]/denominateur

        gamma = [0, 0]
        gamma[0] = sigma[0][0] + sigma[0][1] #proba d'etre dans l'etat 0 a l'instant t
        gamma[1] = sigma[1][0] + sigma[1][1] #proba d'etre dans l'etat 1 a l'instant t

        S[t] = (gamma[0] >= gamma[1])*cl1 + (1-(gamma[0] >= gamma[1]))*cl2
    #on change de methode pour le dernier etat car on n'a pas d'info sur l'etat en T+1
    denominateur_final = alfa[n-1][0] * beta[n-1][0] + alfa[n-1][1] * beta[n-1][1]

    prob_sachant_y1 = alfa[n-1][0] * beta[n-1][0] / denominateur_final
    prob_sachant_y2 = alfa[n-1][1] * beta[n-1][1] / denominateur_final
    S[n-1] = (prob_sachant_y1 >= prob_sachant_y2)*cl1 + (1-(prob_sachant_y1 >= prob_sachant_y2))*cl2
    return S

def MPM_chaines2(Mat_f,n,cl1,cl2,A,p10,p20):
    S = np.zeros(n)
    alfa = scaled_forward2(Mat_f,n,A,p10,p20)
    beta = scaled_backward2(Mat_f, n, A)
    for t in range(n):
        if alfa[t][0]*beta[t][0] >= alfa[t][1]*beta[t][1]:
            S[t] = cl1
        else:
            S[t] = cl2
    return S

def genere_Chaine2(n,cl1,cl2,A,p10,p20):
    chaine = np.zeros(n)
    chaine[0] = tirage_classe2(p10, p20, cl1, cl2)
    for i in range(1, n):
        u = random.random()
        if chaine[i-1] == cl1:
            chaine[i] = (u<=A[0][0])*cl1 + (1-(u<=A[0][0]))*cl2
        else:
            chaine[i] = (u<=A[1][0])*cl1 + (1-(u<=A[1][0]))*cl2
    return chaine

def bruit_gauss2(X, cl1, cl2, m1, sig1, m2, sig2):
    X_cli  = np.zeros(len(X))
    for i in range(len(X)):
        if X[i] == cl1:
            X_cli[i] = np.random.normal(m1, sig1, 1)
        else:
            X_cli[i] =  np.random.normal(m2, sig2, 1)

    return X_cli

def calc_probaprio2(X, cl1, cl2):
    nb_cl1 = 0
    nb_cl2 = 0
    for i in range(len(X)):
        nb_cl1 += (X[i] == cl1)
        nb_cl2 += (X[i] == cl2)
    return (nb_cl1/len(X), nb_cl2/len(X))

def  MAP_MPM2(Y,cl1,cl2,p1,p2,m1,sig1,m2,sig2):
    S = np.zeros(len(Y))
    for i in range(len(Y)):
        y = Y[i]
        if p1*gaussienne(y, m1, sig1) >= p2*gaussienne(y, m2, sig2):
            S[i] = cl1
        else:
            S[i] = cl2
    return S

def erreur_moyenne_MAP(T, X, cl1, cl2, m1, sig1, m2, sig2):
    erreurs = np.zeros(T)
    for i in range(T):
        [p1, p2] = calc_probaprio2(X, cl1, cl2)
        X_bruite = bruit_gauss2(X, cl1, cl2, m1, sig1, m2, sig2)
        S = MAP_MPM2(X_bruite, cl1, cl2, p1, p2, m1, sig1, m2, sig2)
        erreurs[i] = taux_erreur(X, S)
    return erreurs

def erreur_moyenne_BW(T, X, cl1, cl2, m1, sig1, m2, sig2, p10, p20):
    erreurs = np.zeros(T)
    for i in range(T):
        X_bruite = bruit_gauss2(X, cl1, cl2, m1, sig1, m2, sig2)
        n = len(X)
        Mat_f = gauss2(X_bruite, n, m1, sig1, m2, sig2)
        S = MPM_chaines2(Mat_f, n, cl1, cl2, A, p10, p20)
        erreurs[i] = taux_erreur(X, S)
    return erreurs

def moyenne(X):
    sum = 0
    for i in range(len(X)):
        sum += X[i]
    return sum/len(X)

def graphe_chaine_bruitee_segmentee_avec_A(n, A, cl1, cl2, p10, p20, n_bruit):
    X = genere_Chaine2(n, cl1, cl2, A, p10, p20)
    [m1, sig1, m2, sig2] = bruit[n_bruit]
    X_bruite = bruit_gauss2(X, cl1, cl2, m1, sig1, m2, sig2)

    Mat_f = gauss2(X_bruite, n, m1, sig1, m2, sig2)
    S = MPM_chaines2(Mat_f,n,cl1,cl2,A,p10,p20)

    taux_err = taux_erreur(X, S)
    print("taux d'erreur = ", taux_err*100, "%")

    points = np.linspace(0, 100, n)
    plt.plot(points, X, label = 'Chaine de Markov')
    plt.plot(points, X_bruite, color = 'black', label = 'Chaine bruitee')
    plt.scatter(points, S, color = 'red', label = 'Chaine segmentee')
    plt.legend(loc="lower right")
    plt.show()

def  calc_transit_prio2(X,n,cl1,cl2):
    A = [[0, 0], [0, 0]]
    for i in range(n-1):
        if X[i] == cl1:
            if X[i+1] == cl1:
                A[0][0] += 1
            else:
                A[0][1] += 1
        else:
            if X[i+1] == cl1:
                A[1][0] += 1
            else:
                A[1][1] += 1

    sum1 = A[0][0] + A[0][1]
    if sum1 != 0:
        A[0][0] = A[0][0] / sum1
        A[0][1] = A[0][1] / sum1

    sum2 = A[1][0] + A[1][1]
    if sum2 != 0:
        A[1][0] = A[1][0] / sum2
        A[1][1] = A[1][1] / sum2
    return A

def graphe_chaine_bruitee_segmentee_sans_A(n, X, cl1, cl2, n_bruit):
    A = calc_transit_prio2(X, n, cl1, cl2)
    p10 = p20 = 0.5
    [m1, sig1, m2, sig2] = bruit[n_bruit]
    X_bruite = bruit_gauss2(X, cl1, cl2, m1, sig1, m2, sig2)

    Mat_f = gauss2(X_bruite, n, m1, sig1, m2, sig2)
    S = MPM_chaines2(Mat_f,n,cl1,cl2,A,p10,p20)

    taux_err = taux_erreur(X, S)
    print("taux d'erreur = ", taux_err*100, "%")

    points = np.linspace(0, 100, n)
    plt.plot(points, X, label = 'Chaine de Markov')
    plt.plot(points, X_bruite, color = 'black', label = 'Chaine bruitee')
    plt.scatter(points, S, color = 'red', label = 'Chaine segmentee')
    plt.legend(loc="lower right")
    plt.show()

images = ["images_binaires/zebre2.bmp"]
bruit = [[0], [120, 1, 130, 2], [127, 1, 127, 5], [127, 1, 128, 1], [127, 0.1, 128, 0.1],
           [127, 2, 128, 3]]


for n_image in images:
    print(n_image)
    image = plt.imread(n_image)
    plt.imshow(image, cmap = 'gray', vmin = 0, vmax = 255)
    plt.show()
    X = ch.image_to_chain(image)
    n = len(X)
    cl1, cl2 = np.unique(X)
    print("cl1 = ", cl1, ", cl2 = ", cl2)
    A = calc_transit_prio2(X, n, cl1, cl2)
    print("A = ", A)
    p10 = p20 = 0.5
    print("p10 = ", p10, "p20 = ", p20)
    for n_bruit in range(1, 6):
        print("bruit n°", n_bruit)
        [m1, sig1, m2, sig2] = bruit[n_bruit]
        X_bruite = bruit_gauss2(X, cl1, cl2, m1, sig1, m2, sig2)

        noised_image = ch.chain_to_image(X_bruite)
        plt.imshow(noised_image, cmap = 'gray', vmin = 0, vmax = 255)
        plt.show()

        S_gauss = classif_gauss2(X_bruite, cl1, cl2, m1, sig1, m2, sig2)

        [p1, p2] = calc_probaprio2(X, cl1, cl2)
        print("p1 = ", p1, ", p2 = ", p2)
        S_MAP = MAP_MPM2(X_bruite, cl1, cl2, p1, p2, m1, sig1, m2, sig2)

        Mat_f = gauss2(X_bruite, n, m1, sig1, m2, sig2)
        S_BW = MPM_chaines2(Mat_f, n, cl1, cl2, A, p10, p20)

        segmented_image = ch.chain_to_image(S_gauss)
        plt.imshow(segmented_image, cmap = 'gray', vmin = 0, vmax = 255)
        plt.show()

        segmented_image = ch.chain_to_image(S_MAP)
        plt.imshow(segmented_image, cmap = 'gray', vmin = 0, vmax = 255)
        plt.show()

        segmented_image = ch.chain_to_image(S_BW)
        plt.imshow(segmented_image, cmap = 'gray', vmin = 0, vmax = 255)
        plt.show()







