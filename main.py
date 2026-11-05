from scipy.io import loadmat
from sklearn.cluster import KMeans, BisectingKMeans, AgglomerativeClustering
from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
from Codificacion.register import register, searchCents, searchTree,searchSCAN
from os import path,listdir
from Codificacion.metrics import euclideanDistance,hammingDistance
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,silhouette_score

COD='gabor'
CLST='hdbscan'

def joinTable(clusters:HDBSCAN|BisectingKMeans|KMeans|AgglomerativeClustering,ids:list,temps:list):
	table=[]
	for ind in range(len(ids)):
		table.append([clusters.labels_[ind],ids[ind],temps[ind]])
		#print(clusters.labels_[ind],ids[ind])
	print(clusters.labels_)
	table.sort(key= lambda x:x[0])
	return table


"""
	print("Registrando plantillas...")
#leer imagenes
	entrys=listdir("Images/")
	resgistered=listdir(path.join("Templates",COD))
	for im in entrys:
	#codificar imagenes
		if im[:-4]+'.mat'not in resgistered:
			register(path.join('Images',im),COD,im[:-4])
	
#leer plantillas
"""

def registerDir(dir:str,code:str):
	"""
	Registra 1 de las imagenes de cada individuo y retorna el path de cada imagen registrada
	"""
	print(f"Registrando con codificacion {code}")
	ids=listdir(dir)
	paths=[]
	y_true=[]
	registered=listdir(path.join("Templates",COD))
	for i in ids:
		ims=listdir(f"{dir}/{i}/")
		paths.append(f"{dir}/{i}/{ims[0]}")
		y_true.append(i)
		if ims[0][:-4]+'.mat' not in registered:
			register(f"{dir}/{i}/{ims[0]}",code,i)
	return paths,y_true

def getTemplates(cod:str):
	entrys=listdir(path.join("Templates",COD))
	
	temps=[]
	ids=[]
	
	for temp in entrys:
		template=loadmat(path.join("Templates",COD,temp))
		ids.append(''.join(template['id']))
		temps.append(template['template'].flatten())
	return temps,ids

if __name__=='__main__':
	dist=euclideanDistance
	if COD=='gabor':
		dist=hammingDistance
	print("Registrando...")

	regs,y_true =registerDir("C:\\Users\\wings\\Desktop\\CASIA\\TEST",COD)

	print("Leyendo plantillas...")
	temps,ids=getTemplates(COD)

	preds=[]
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
			table=joinTable(tree,ids,temps)
			print("Evaluando...")
			for r in regs:
				id,im,dis,time,comps=searchTree(r,COD,tree,table,euclideanDistance)
				preds.append(id)
			print(f"Resultados de {COD} indexado con {CLST}")
			print(f"Acurracy:{accuracy_score(y_true,preds)}")
			print(f"Prescision:{precision_score(y_true,preds,average='micro')}")
			print(f"F1 score:{f1_score(y_true,preds,average='micro')}")
			print(f"Recall:{recall_score(y_true,preds,average='micro')}")
			print(f"Silhouette score:{silhouette_score(temps,tree.labels_)}")
		case 'avg':
			tree=AgglomerativeClustering(n_clusters=3,linkage='average').fit(temps)
			table=joinTable(tree,ids,temps)
			print("Evaluando...")
			for r in regs:
				id,im,dis,time,comps=searchTree(r,COD,tree,table,euclideanDistance)
				preds.append(id)
			print(f"Resultados de {COD} indexado con {CLST}")
			print(f"Acurracy:{accuracy_score(y_true,preds)}")
			print(f"Prescision:{precision_score(y_true,preds,average='micro')}")
			print(f"F1 score:{f1_score(y_true,preds,average='micro')}")
			print(f"Recall:{recall_score(y_true,preds,average='micro')}")
			print(f"Silhouette score:{silhouette_score(temps,tree.labels_)}")
		case 'hdbscan':
			db=HDBSCAN(min_cluster_size=3,max_cluster_size=35, store_centers='centroid',copy=False).fit(temps)
			table= joinTable(db,ids,temps)
			for r in regs:
				id,im,dis,time,comps=searchSCAN(r,COD,db,table,dist)
				preds.append(id)
			print(f"Resultados de {COD} indexado con {CLST}")
			print(f"Acurracy:{accuracy_score(y_true,preds)}")
			print(f"Prescision:{precision_score(y_true,preds,average='micro')}")
			print(f"F1 score:{f1_score(y_true,preds,average='micro')}")
			print(f"Recall:{recall_score(y_true,preds,average='micro')}")
			print(f"Silhouette score:{silhouette_score(temps,db.labels_)}")
		