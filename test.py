from scipy.io import loadmat
from sklearn.cluster import KMeans, BisectingKMeans, AgglomerativeClustering
from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
from Codificacion.register import register, searchCents, searchTree,searchSCAN
from os import path,listdir
from Codificacion.metrics import euclideanDistance,hammingDistance
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score,silhouette_score
import numpy as np

cods=['dct','dwt','gabor','pca']
clsts=['k-means','b-means','k-means++','avg','ward','hdbscan']

def joinTable(clusters:HDBSCAN|BisectingKMeans|KMeans|AgglomerativeClustering,ids:list,temps:list):
	table=[]
	for ind in range(len(ids)):
		table.append([clusters.labels_[ind],ids[ind],temps[ind]])
		#print(clusters.labels_[ind],ids[ind])
	table.sort(key= lambda x:x[0])
	return table

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

def test(COD:str,CLST:str):
	dist=euclideanDistance
	if COD=='gabor':
		dist=hammingDistance
		#leer imagenes
	regs,y_true =registerDir("C:\\Users\\wings\\Desktop\\CASIA\\TEST",COD)
		#leer plantillas
	print("Leyendo plantillas...")
	
	temps,ids=getTemplates(COD)

	t_scores=[]	
	preds=[]
	#crear clusters
	print("Generando clusteres...")
	match CLST:
		case 'k-means':
			clusters=KMeans(n_clusters=3,random_state=0).fit(temps)
			table=joinTable(clusters,ids,temps)
			print("Evaluando...")
			for r in regs:
				id,im,dis,time,comps=searchCents(r,COD,clusters,table,dist)
				t_scores.append(time)
				preds.append(id)
			sil=silhouette_score(temps,clusters.labels_)

		case 'b-means':
			clusters=BisectingKMeans(n_clusters=3,random_state=0).fit(temps)
			table=joinTable(clusters,ids,temps)
			print("Evaluando...")
			for r in regs:
				id,im,dis,time,comps=searchCents(r,COD,clusters,table,dist)
				t_scores.append(time)
				preds.append(id)
			sil=silhouette_score(temps,clusters.labels_)

		case 'k-means++':
			clusters=KMeans(n_clusters=3,random_state=0,init='k-means++').fit(temps)
			table=joinTable(clusters,ids,temps)
			print("Evaluando...")
			for r in regs:
				id,im,dis,time,comps=searchCents(r,COD,clusters,table,dist)
				t_scores.append(time)
				preds.append(id)
			sil=silhouette_score(temps,clusters.labels_)
		
		case 'ward':
			tree=AgglomerativeClustering(n_clusters=3,linkage='ward').fit(temps)
			table=joinTable(tree,ids,temps)
			print("Evaluando...")
			for r in regs:
				id,im,dis,time,comps=searchTree(r,COD,tree,table,euclideanDistance)
				t_scores.append(time)
				preds.append(id)
			sil=silhouette_score(temps,tree.labels_)

		case 'avg':
			tree=AgglomerativeClustering(n_clusters=3,linkage='average').fit(temps)
			table=joinTable(tree,ids,temps)
			print("Evaluando...")
			for r in regs:
				id,im,dis,time,comps=searchTree(r,COD,tree,table,euclideanDistance)
				t_scores.append(time)
				preds.append(id)
			sil=silhouette_score(temps,tree.labels_)
			
		case 'hdbscan':
			db=HDBSCAN(min_cluster_size=3,max_cluster_size=35, store_centers='centroid',copy=False).fit(temps)
			table= joinTable(db,ids,temps)
			print("Evaluando...")
			for r in regs:
				id,im,dis,time,comps=searchSCAN(r,COD,db,table,dist)
				t_scores.append(time)
				preds.append(id)
			sil=silhouette_score(temps,db.labels_)
	avg_t=np.sum(t_scores)/len(t_scores)
	return y_true,preds,sil,avg_t

if __name__=='__main__':
	with open("results_mi_pcr.txt", "w") as f:
		f.write(f"Cluster_type\tCod\tAcc\tPres\tF1\tRec\tAvgT\tSil\n")
		f.close()

	for COD in cods:
		for CLST in clsts:
			y_true,preds,sil,avg_t=test(COD,CLST)
			acc=accuracy_score(y_true,preds)
			pres=precision_score(y_true,preds,average='micro')
			f1=f1_score(y_true,preds,average='micro')
			rec=recall_score(y_true,preds,average='micro')
			print(f"Resultados de {COD} indexado con {CLST}")
			print(f"Acurracy:{acc}")
			print(f"Prescision:{pres}")
			print(f"F1 score:{f1}")
			print(f"Recall:{rec}")
			print(f"Tiempo medio de respuesta:{avg_t}")
			print(f"Silhouette score:{sil}")
			f=open("results.txt", "a")
			f.write("{:}\t{:}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(CLST,COD,acc,pres,f1,rec,avg_t,sil))
			f.close()
