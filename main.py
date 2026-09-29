from scipy.io import loadmat
from sklearn.cluster import KMeans, BisectingKMeans, AgglomerativeClustering
from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
from Codificacion.register import register, searchCents, searchTree,searchSCAN
from os import path,listdir
from Codificacion.metrics import euclideanDistance,hammingDistance
import numpy as np

COD='pca'
CLST='ward'

def joinTable(clusters:HDBSCAN|BisectingKMeans|KMeans|AgglomerativeClustering,ids:list):
	table=[]
	for ind in range(len(ids)):
		table.append([clusters.labels_[ind],ids[ind],temps[ind]])
	table.sort()
	return table

if __name__=='__main__':
	print("Registrando plantillas...")
#leer imagenes
	entrys=listdir("Images/")
	resgistered=listdir(path.join("Templates",COD))
	for im in entrys:
	#codificar imagenes
		if im[:-4]+'.mat'not in resgistered:
			register(path.join('Images',im),COD,im[:-4])

#leer plantillas
	print("Leyendo plantillas...")

	entrys=listdir(path.join("Templates",COD))

	temps=[]
	ids=[]

	for temp in entrys:
		template=loadmat(path.join("Templates",COD,temp))
		ids.append(''.join(template['id']))
		temps.append(template['template'].flatten())

#crear clusters
	print("Generando clusteres...")
	match CLST:
		case 'k-means':
			clusters=KMeans(n_clusters=3,random_state=0).fit(temps)
		case 'b-means':
			clusters=BisectingKMeans(n_clusters=3,random_state=0).fit(temps)
		case 'k-means++':
			clusters=KMeans(n_clusters=3,random_state=0,init='k-means++').fit(temps)
		case 'ward':
			tree=AgglomerativeClustering(n_clusters=3,linkage='ward').fit(temps)
		case 'avg':
			tree=AgglomerativeClustering(n_clusters=3,linkage='average').fit(temps)
		case 'dbscan':
			db=HDBSCAN(min_samples=1, store_centers='centroid',copy=False).fit(temps)

	# Busquedas con centroides
	if CLST in ['k-means','b-means','k-means++']:
		table=joinTable(clusters,ids)
		res=searchCents("Images\\jpD2.bmp",COD,clusters,table,euclideanDistance)
		print(res)

	#Busqueda en arbol
	if CLST in ['ward','avg']:
		table=joinTable(tree,ids)
		res =searchTree("Images\\100.bmp",COD,tree,table,euclideanDistance)
		print(res)

	#Busqueda en DBSCAN
	if CLST=='hdbscan':
		table= joinTable(db,ids)
		print(table[0])
		res= searchSCAN("Images\\100.bmp",COD,db,table,euclideanDistance)
		print(res)