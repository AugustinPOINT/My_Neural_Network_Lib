import lib_neuron_network_v2 as lnn
import numpy as np
import time
import os


def main():

	affichage = False
	duree = 0

	_NETWORKS_ = [] #Liste de réseaux
	nb_networks = 0

	os.system('cls')
	while True:
		print("""
	0. Quitter
	1. Créer un réseau de neurones complètement connecté
	2. Choix de réseau
	""")
		print("-->",end=' ')
		choix = int(input())
		print("\n")
		if(choix == 0):
			break
		elif(choix == 1):
			print("Entrez le nombre de couches total du réseau : ", end=' ')
			nb_couches = int(input())
			tab_couches = [0 for i in range(nb_couches)]
			tab_activation = [None for i in range(nb_couches-1)]
			print("   Entrez le nombre d'entrées du réseau : ", end=' ')
			tab_couches[0] = int(input())
			for couche in range(1,nb_couches):
				print(f"   Entrez le nombre de neurones de la couche {couche} : ", end=' ')
				tab_couches[couche] = int(input())
			print(f"Entrez les fonctions d'activation des différentes couches : 1:ReLU, 2:LeakyReLU 3:Tanh, 4:Sigmoid")
			corr = ["ReLU", "LeakyReLU","Tanh", "Sigmoid"]
			for couche in range(1,nb_couches):
				while True:
					print(f"   Couche {couche} : ", end=' ')
					choix_act = int(input())
					if(choix_act > 0 and choix_act <= 4):
						break
					else : print(f"    Erreur")
				tab_activation[couche-1] = corr[choix_act-1]
			print("Entrez le coefficient d'apprentissage alpha : ", end=' ')
			alpha = float(input())
			_NETWORKS_.append(lnn.Neuronal_Network(tab_couches, tab_activation, alpha))
			nb_networks += 1
			os.system('cls')
		elif(choix == 2):
			if(nb_networks == 0):
				os.system('cls')
				print("Aucun réseau n'a été créé")
				continue
			while True:
				print("Sur quel réseau voulez-vous travailler ?")
				for reseau in range(nb_networks):
					print(f"   - Reseau {reseau} : ")
					print(f"       Agencement : {_NETWORKS_[reseau].couches}")
					print(f"       Activation : {_NETWORKS_[reseau].activation_repartition}")
					print(f"       Coeff d'apprentissage : {_NETWORKS_[reseau].alpha}")
				print("-->",end=' ')
				choix_net = int(input())
				if(choix_net >= 0 and choix_net <= nb_networks):
					break
			os.system('cls')
			while True:
				print(f"Réseau de Neurones n°{choix_net}")
				print("""
	0. Retour
	1. Charger le fichier de training
	2. Afficher les prédictions sur les Training examples
	3. Entrainer le réseau avec les Training examples
	4. Changer le coefficient alpha
	5. Afficher les poids du réseau
	6. Afficher le taux de prédiction sur les Training examples
	7. Réinitialiser le réseau avec des valeurs aléatoires
	""")
				print("-->",end=' ')
				choix = int(input())
				print("\n")
				if(choix == 0):
					break
				elif(choix == 1):
					print("Entrez le chemin du fichier par rapport au dossier courant : ", end=' ')
					path = input()
					ret = _NETWORKS_[choix_net].set_training_data_by_file(path)
					if(ret == -1):
						os.system('cls')
						print("Erreur d'ouverture du fichier")
						continue
					if(ret == -2):
						os.system('cls')
						print("Erreur de format des données")
						continue
					os.system('cls')
					print("Fichier chargé")
					print(_NETWORKS_[choix_net].training_data_X)
					print(_NETWORKS_[choix_net].training_data_Y)
				elif(choix == 2):
					pred, err, err, J = _NETWORKS_[choix_net].predict()
					os.system('cls')
					print(f" Prédiction : {pred[len(_NETWORKS_[choix_net].couches)-1]}")
					print(f" Fonction de coût : {J}")
				elif(choix == 3):
					print("Entrez le nombre d'itérations voulu : ", end='')
					iterr = int(input())
					_NETWORKS_[choix_net].train(iterr)
					#os.system('cls')
					print(f"Réseau correctement entrainé sur {iterr} itérations")
				elif(choix == 4):
					print("Entrez un nouveau coefficient d'apprentissage alpha : ", end=' ')
					alpha = float(input())
					_NETWORKS_[choix_net].alpha = alpha
					os.system('cls')
					print("Coefficient alpha changé")
				elif(choix == 5):
					os.system('cls')
					print("Ws : ")
					for i in range(len(_NETWORKS_[choix_net].couches)-1):
						print(f"W couche {i+1} : {_NETWORKS_[choix_net].weights[i][0]}")
						print(f"b couche {i+1} : {_NETWORKS_[choix_net].weights[i][1]}")
				elif(choix == 6):
					pourc = _NETWORKS_[choix_net].taux_prediction()*100
					os.system('cls')
					print(f"Taux de prédiction sur les Training examples : {pourc}")
				elif(choix == 7):
					_NETWORKS_[choix_net].weights = _NETWORKS_[choix_net].compute_weights_randomly(0.01)
					os.system('cls')
					print("Poids correctement réinitialisés")
			os.system('cls')
		elif(choix == 3):
			os.system('cls')
			continue

	


if __name__ == "__main__":
    main()
