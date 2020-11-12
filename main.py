import random
import operator
import math
import numpy
import numpy as np
import csv
import datetime
import os
import time

# deap
# pip install deap
from deap import base, creator, gp, tools, algorithms
# keras
# pip install keras
import tensorflow
import keras
from keras.models import Sequential, clone_model
import keras.utils
from keras.layers import Dense, Input
from keras.layers.merge import concatenate
# pip install tensorflow
# split data with sklearn library
# pip install sklearn
from sklearn.model_selection import train_test_split
import datetime
from datetime import date


def log_title_console(title):
    print("\n#####{}#####".format("#" * len(title)))
    print("#   ", title, "   #")
    print("#####{}#####\n".format("#" * len(title)))


################################################################################

log_title_console("DATASET")

TAILLE_POP = 10
NOMBRE_LIGNES = 300
NOMBRE_VARIABLE = 14

# GPU Kernel Performance Dataset (kaggle.com)
# Link : https://www.kaggle.com/rupals/gpu-runtime?select=sgemm_product.csv
dataset = []
data_file = open("sgemm_product.csv")
read_data = csv.reader(data_file, delimiter=",")
for row in read_data:
    dataset.append(row)
dataset = dataset[1:]

# Input : x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 y1 y2 y3 y4
# Considérons les variables y de la manière suivante :
# 1- On réduit la dimension à 1 tel que y = sum(y1, y2, y3, y4)
mean_y = []
for i in range(NOMBRE_LIGNES):
    for j in range(len(dataset[i])):
        if j < NOMBRE_VARIABLE:
            dataset[i][j] = float(dataset[i][j])
        elif j == NOMBRE_VARIABLE:
            for elem in dataset[i][-4:]:
                dataset[i][dataset[i].index(elem)] = float(elem)
            new_y = round(sum(dataset[i][-4:]), 2)
            dataset[i][j] = new_y
            mean_y.append(new_y)
    # Les derniers 'y' doivent étre effacé
    dataset[i] = dataset[i][:-3]
dataset = dataset[:NOMBRE_LIGNES]

# 2 - On transforme y tel que :
# y = 0 si y < moyenne des y observés, 1 sinon
# L'objectif est d'obtenir une classification binaire
X = []
y = []
for tuple in dataset:
    X.append(tuple[:14])
    if tuple[-1] >= np.mean(mean_y):
        y.append(1)
    else:
        y.append(0)
X = np.array(X)
y = np.array(y)

# On split nos données d'entraînement, de test et de validation tel que:
# -> inspiré du cours "COMPUTER VISION"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)
# Test rapide
print("Dataset:              split data...")
print('> Initial size:      ', len(dataset))
print('> training samples:  ', len(X_train))
print('> testing samples:   ', len(X_test))
print('> validation samples:', len(X_val))
if len(X_train) + len(X_test) + len(X_val) == len(dataset):
    print("                      OK!")
else:
    print("                      KO!")

################################################################################

log_title_console("GENERATION INDIVIDU")


def generate_weight_matrix(number_hidden_layers, number_neurons):
    weight_matrix = []
    # On défini maintenant un vecteur de poids pour chaques neurones de chaques couches
    for i in range(number_hidden_layers):
        vector = []
        # Le dernier vecteur de poids
        if i == number_hidden_layers-1:
            for k in range(number_neurons[i]):
                vector.append(random.random() if random.random() < 0.7 else 0)
        # Les autres
        else:
            for j in range(number_neurons[i] * number_neurons[i+1]):
                # Pour casser le système 'fully connected' du réseau, on définit certains poids à 0
                vector.append(random.random() if random.random() < 0.7 else 0)
        # On l'ajoute à notre liste de poids
        weight_matrix.append(vector)
    return weight_matrix


def generate_individual():
    # On retourne le tableau suivant:
    # - un tableau : [nombre de couches, [nombres de neurones dans chaques couches]] -> neural_network_config
    # - un vecteur de poids pour chaques couches -> weight_vector
    res = []
    neural_network_config = []
    # On choisit un nombre aléatoire de couches cachées
    number_hidden_layers = int(random.uniform(1, 10))
    # On choisit un nombre aléatoire de neurones dans chaques couches
    number_neurons = []
    for i in range(number_hidden_layers):
        number_neurons.append(int(random.uniform(1, NOMBRE_VARIABLE)))
    # On ajoute à la liste
    neural_network_config.append(number_hidden_layers)
    neural_network_config.append(number_neurons)
    # On l'ajoute au résultat final
    res.append(neural_network_config)
    weight_vector = generate_weight_matrix(number_hidden_layers, number_neurons)
    # On l'ajoute au résultat
    res.append(weight_vector)
    return res


print("Generate model:")
example = generate_individual()
print('> model:\n', example)

################################################################################

log_title_console("GENERATION POPULATION")

# On crée notre population tel que :
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Population
pop = toolbox.population(n=TAILLE_POP)

print("Generate pop")
print("> len(pop):  ", len(pop))
print("> pop[0]:    ", pop[0])

################################################################################

log_title_console("OPERATIONS WITH DEAP")

# Création du fichier de sortie
output = open(str(date.today()) + "_MIAGE_FELIN_AE_res.txt", "w")
name = output.name

output.write("> Auteur:    Rémi FELIN\n")
output.write("> TP6:       Algorithmes évolutionnaires\n")
output.write("> Date:      {}\n".format(datetime.datetime.now()))


def build_model(input):
    # On crée notre réseau de neurones ici :
    model = Sequential()
    # Nos entrées fixes :
    model.add(Dense(NOMBRE_VARIABLE, input_dim=NOMBRE_VARIABLE, activation='sigmoid', name="input"))
    # Nos couches cachées
    for i in range(input[0][0]):
        model.add(Dense(input[0][1][i], activation='sigmoid', name="Couche_" + str(i + 1)))
    # On ajoute notre sortie
    model.add(Dense(1, activation='sigmoid', name="output"))
    # model.summary()
    return model


def fitness(individual):
    # Notre modèle
    model = build_model(individual)
    # On compile notre modèle
    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['mse'])
    # On l'entraine avec le jeu de données (avec peu d'epoch et un petit batch)
    model.fit(X_train,
              y_train,
              validation_data=(X_val, y_val),
              epochs=10,
              batch_size=5,
              verbose=0)
    # On évalue l'erreur quadratique du modèle
    score = model.evaluate(X_test, y_test, verbose=1)
    # On applique la formule et on retourne le résultat
    return 1 / score[0] + 1,


def crossover(individual1, individual2):
    # On teste aussi si les 2 individus ont la même dimension
    if individual1[0][0] == individual2[0][0]:
        # On applique un crossover de type merge entre 2 modèles
        for i in range(len(individual1[0][1])):
            # On ajoute les élèments d'un modèle dans l'autre
            individual1[0][1][i] = individual1[0][1][i] + individual2[0][1][i]
        # On regénère les poids
        individual1[1] = generate_weight_matrix(individual1[0][0], individual1[0][1])
        return individual1, individual2
    else:
        # On retourne les individus sans rien faire
        return individual1, individual2


def mutate(individual):
    # On ajoute un noeud dans la dernière couche
    # de plus, on divise son poid avec le noeud adjacent
    # > On sélectionne la dernière couche et on lui ajoute un neurone
    individual[0][1][-1] = individual[0][1][-1] + 1
    # On met à jour les 2 poids adjacent en les divisant par 2
    # > On divise par 2 le dernier neurone :
    individual[1][-1][-1] = individual[1][-1][-1] / 2
    # > On ajoute le dernier poid déjà divisé au préalable
    individual[1][-1].append(individual[1][-1][-1])
    return individual,


toolbox.register("mate", crossover)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", fitness)

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
mstats = tools.MultiStatistics(fitness=stats_fit)
mstats.register("avg", numpy.mean)
mstats.register("std", numpy.std)
mstats.register("min", numpy.min)
mstats.register("max", numpy.max)

NGEN = 10
CXPB = 0.4
MUTPB = 0.2

# TESTING
output.write("\n> Starting ...\n")

hof = tools.HallOfFame(3)
pop, log = algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, stats=mstats,
                               halloffame=hof, verbose=True)

output.write(str(log))

output.write("\n\n> End !\n")

output.write("\n\n[RESULT] - 3 best individuals :\n")
for res in hof:
    output.write("\n##########################################################")
    output.write("\n> individual: " + str(res))
    output.write("\n\n> summary")
    list = []
    build_model(res).summary(print_fn=lambda x: list.append(x))
    model_summary = "\n".join(list)
    output.write("\n\n" + str(model_summary))
    output.write("\n\n> fitness: " + str(res.fitness))
    output.write("\n\n##########################################################\n")

output.close()

if open(name, "r") is not None:
    print("\n[END] Le fichier", name, "a bien été créé")
