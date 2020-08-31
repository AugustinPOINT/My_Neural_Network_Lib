import math as m
from pprint import pprint
import numpy as np
import time
import os
LEAKY_RELU_COEFF = 0.001

def ReLU(array : np.ndarray):
	condlist = [array <= 0]
	funclist = [lambda x: 0, lambda x: x]
	Z = np.piecewise(array, condlist, funclist)
	return Z

def LeakyReLU(array : np.ndarray):
	condlist = [array <= 0]
	funclist = [lambda x: LEAKY_RELU_COEFF*x, lambda x: x]
	Z = np.piecewise(array, condlist, funclist)
	return Z

def Sigmoid(normale : bool, array : np.ndarray):	
	return 1/(1+np.exp(-array))

def Tanh(normale : bool, array : np.ndarray):
	return (np.exp(array)-np.exp(-array))/(np.exp(array)+np.exp(-array))


def dReLU(array : np.ndarray):
	condlist = [array <= 0]
	funclist = [lambda x: 0, lambda x: 1]
	Z = np.piecewise(array, condlist, funclist)
	return Z

def dLeakyReLU(array : np.ndarray):
	condlist = [array <= 0]
	funclist = [lambda x: LEAKY_RELU_COEFF, lambda x: 1]
	Z = np.piecewise(array, condlist, funclist)
	return Z

def dSigmoid(normale : bool, array : np.ndarray):
	if normale : #Si la dérivée est calculée à partir de z
		sig = 1/(1+np.exp(-array))
		Z = sig*(1-sig)
	else: #Si la dérivée est calculée à partir de Sigmoid(z)
		Z = array*(1-array)
	return Z

def dTanh(normale : bool, array : np.ndarray):
	if normale : #Si la dérivée est calculée à partir de z
		sig = (np.exp(array)-np.exp(-array))/(np.exp(array)+np.exp(-array))
		Z = 1-sig*sig
	else: #Si la dérivée est calculée à partir de Tanh(z)
		Z = 1-array*array
	return Z



class BadDepthException(Exception):
    pass


class Neuronal_Network:
	def __init__(self,couches : list, activation_repartition : list, alpha : float):
		assert len(couches) == len(activation_repartition)+1 #La répartition des fonctions d'activation, se fait en en donnant une par couche
		self.couches = couches #Nombre de couches en comptant les entrées comme la couche 0
		self.activation_repartition = activation_repartition
		self.nb_arcs = self.compute_nb_arcs() #Nombre d'arcs en comptant les entrées
		self.nb_units = self.compute_nb_units() #Nombre total d'unitées (sans compter les entrées)
		self.nb_entrees = couches[0] #Le nombre d'entrées est le contenu de la couche 0
		self.weights = self.compute_weights_randomly(0.01)
		self.training_data_X = None
		self.training_data_Y = None
		self.alpha = alpha

	def compute_nb_arcs(self):
		res = 0
		for couche in range(1,len(self.couches)):
			res += self.couches[couche-1] * self.couches[couche]
		return res

	def compute_nb_units(self):
		res = self.couches[1]
		for couche in range(2, len(self.couches)):
			res += self.couches[couche]
		return res

	def compute_weights_randomly(self, scale : int):
		weights = [None for i in range(len(self.couches)-1)]
		n0 = self.nb_entrees #Dimension axe 0 (vertical)
		for couche in range(1, len(self.couches)):
			n1 = n0
			n0 = self.couches[couche]
			if(self.activation_repartition[couche-1] == "ReLU"):
				weights_couche_i = np.random.randn(n0, n1)
			elif(self.activation_repartition[couche-1] == "LeakyReLU"):
				weights_couche_i = np.random.randn(n0, n1)
			elif(self.activation_repartition[couche-1] == "Tanh"):
				weights_couche_i = np.random.randn(n0, n1) * scale
			elif(self.activation_repartition[couche-1] == "Sigmoid"):
				weights_couche_i = np.random.randn(n0, n1) * scale
			biais_couche_i = np.zeros((n0,1))
			weights[couche-1] = [weights_couche_i, biais_couche_i]
		return weights

	def set_training_data_by_file(self, path : str):
		try:
			with open(os.path.dirname(os.path.realpath(__file__)) + "\\" + path, 'r') as file:
				depth, nb_data = np.array(file.readline().split(',')).astype(int)
				if(depth != self.nb_entrees):
					raise BadDepthException()
				self.training_data_X = np.empty([self.nb_entrees, nb_data], dtype = float)
				self.training_data_Y = np.empty([1,nb_data], dtype = float)
				for indice in range(self.nb_entrees):
					data = np.array(file.readline().split(' '))
					t_ex = data.astype(int)
					self.training_data_X[indice] = t_ex
				data = np.array(file.readline().split(' '))
				values = data.astype(int)
				if(values.shape[0] != nb_data):
					raise BadDepthException()
				self.training_data_Y = values
				self.scale_data()
		except FileNotFoundError:
			return -1
		except BadDepthException:
			return -2
		return 0

	def scale_data(self):
 		max = np.amax(self.training_data_X[0])
 		for i in range(1,self.training_data_X.shape[0]):
 			maxline = np.amax(self.training_data_X[i])
 			if(maxline > max):
 				max = maxline
 		if(max == 0):
 			max = 1
 		self.training_data_X /= max
 		return 0

	def set_training_data_by_matrix(self, X : np.ndarray, Y : np.ndarray):
		if(X.shape[0] != self.nb_entrees or X.shape[1] != Y.shape[0]):
			raise BadDepthException()
		self.training_data_Y = X
		self.training_data_Y = Y
		return 0

	def predict(self):
		m = self.training_data_X.shape[1]
		A = np.array([None for i in range(len(self.couches))], dtype = object)
		Z = np.array([None for i in range(len(self.couches)-1)], dtype = object)
		B = None
		A[0] = self.training_data_X
		for couche in range(1, len(self.couches)):
			Z[couche-1] = np.dot(self.weights[couche-1][0], A[couche-1]) + self.weights[couche-1][1]
			if(self.activation_repartition[couche-1] == "ReLU"):
				A[couche] = ReLU(Z[couche-1])
			elif(self.activation_repartition[couche-1] == "LeakyReLU"):
				A[couche] = LeakyReLU(Z[couche-1])
			elif(self.activation_repartition[couche-1] == "Tanh"):
				A[couche] = Tanh(True, Z[couche-1])
			elif(self.activation_repartition[couche-1] == "Sigmoid"):
				A[couche] = Sigmoid(True, Z[couche-1])
				if(couche == len(self.couches)-1): #Si dernière couche :
					B = Sigmoid(True, -Z[couche-1])
		L = np.negative(np.add((self.training_data_Y * np.log(A[len(self.couches)-1])) , ((1-self.training_data_Y) * np.log(B))))
		J = np.sum(L)/m
		return A, Z, B, J

	def train(self, nb_iter : int):
		for iteration in range(nb_iter):
			m = self.training_data_X.shape[1]
			A, Z, B, J = self.predict()
			dA = ((A[len(self.couches)-1] - self.training_data_Y)/(A[len(self.couches)-1]*(1-A[len(self.couches)-1])))
			if(self.activation_repartition[len(self.couches)-2] == "ReLU"):
				dZ = dA * dReLU(Z[len(self.couches)-2])
			elif(self.activation_repartition[len(self.couches)-2] == "LeakyReLU"):
				dZ = dA * dLeakyReLU(Z[len(self.couches)-2])
			elif(self.activation_repartition[len(self.couches)-2] == "Tanh"):
				dZ = dA * dTanh(True, Z[len(self.couches)-2])
			elif(self.activation_repartition[len(self.couches)-2] == "Sigmoid"):
				dZ = A[len(self.couches)-1] - self.training_data_Y
			dW = np.dot(dZ, A[len(self.couches)-2].T)/m #Cas particulier de la dernière couche
			db = np.sum(dZ)/m
			self.weights[len(self.couches)-2][0].__isub__(self.alpha * dW)
			self.weights[len(self.couches)-2][1].__isub__(self.alpha * db)
			for couche in reversed(range(1, len(self.couches)-1)): #Pour les autres couches :
				dZ = np.dot(np.array(self.weights[couche][0], dtype = float).T, dZ)
				if(self.activation_repartition[couche-1] == "ReLU"): #Calcul du nouveau dZ selon la fonction d'activation choisie
					dZ = dZ * dReLU(Z[couche-1])
				elif(self.activation_repartition[couche-1] == "LeakyReLU"):
					dZ = dZ * dLeakyReLU(Z[couche-1])
				elif(self.activation_repartition[couche-1] == "Tanh"):
					dZ = dZ * dTanh(True, Z[couche-1])
				elif(self.activation_repartition[couche-1] == "Sigmoid"):
					dZ = dZ * dSigmoid(True, Z[couche-1])
				dW = np.dot(dZ, A[couche-1].T)/m #Cas général des autres couches
				db = np.sum(dZ)/m
				self.weights[couche-1][0].__isub__(self.alpha * dW)
				self.weights[couche-1][1].__isub__(self.alpha * db)
			if(iteration%100 == 0):
				print(f"Itération {iteration} : J = {J}")
		return J

	def taux_prediction(self):
		nbre_correct = 0
		m = self.training_data_X.shape[1]
		A = self.predict()[0][len(self.couches)-1]
		pred = A[0]
		for i in range(m):
			condlist = [pred <= 0.5]
			funclist = [lambda x: 0, lambda x: 1]
			pred = np.piecewise(pred, condlist, funclist)
			if(pred[i] == self.training_data_Y[i]):
				nbre_correct += 1
		return nbre_correct/m

