from scipy.io import loadmat
from sklearn.cluster import KMeans, BisectingKMeans, AgglomerativeClustering
from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
from Codificacion.register import register, searchCents, searchTree,searchSCAN
from os import path,listdir
from Codificacion.metrics import euclideanDistance,hammingDistance
import numpy as np

COD='gabor'
CLST='avg'

def joinTable(clusters:HDBSCAN|BisectingKMeans|KMeans|AgglomerativeClustering,ids:list):
	table=[]
	for ind in range(len(ids)):
		table.append([clusters.labels_[ind],ids[ind],temps[ind]])
		print(clusters.labels_[ind],ids[ind])
	table.sort()
	return table

if __name__=='__main__':
	dist=euclideanDistance
	if COD=='gabor':
		dist=hammingDistance

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
			table=joinTable(clusters,ids)
			res=searchCents("Images\\jpD2.bmp",COD,clusters,table,dist)
			print(res)

		case 'b-means':
			clusters=BisectingKMeans(n_clusters=3,random_state=0).fit(temps)
			table=joinTable(clusters,ids)
			res=searchCents("Images\\jpD2.bmp",COD,clusters,table,dist)
			print(res)
		
		case 'k-means++':
			clusters=KMeans(n_clusters=3,random_state=0,init='k-means++').fit(temps)
			table=joinTable(clusters,ids)
			res=searchCents("Images\\jpD2.bmp",COD,clusters,table,dist)
			print(res)
		
		case 'ward':
			tree=AgglomerativeClustering(n_clusters=3,linkage='ward').fit(temps)
			table=joinTable(tree,ids)
			res =searchTree("Images\\100.bmp",COD,tree,table,dist)
			print(res)

		case 'avg':
			tree=AgglomerativeClustering(n_clusters=3,linkage='average').fit(temps)
			table=joinTable(tree,ids)
			res =searchTree("Images\\20322376-50.bmp",COD,tree,table,dist)
			print(res)
		
		case 'hdbscan':
			db=HDBSCAN(min_samples=5, store_centers='centroid',copy=False).fit(temps)
			table= joinTable(db,ids)
			res= searchSCAN("Images\\gaboD2.bmp",COD,db,table,dist)
			print(res)