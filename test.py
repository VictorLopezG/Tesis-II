from scipy.io import loadmat
from sklearn.cluster import KMeans, BisectingKMeans, AgglomerativeClustering
from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
from Codificacion.register import register, searchCents, searchTree,searchSCAN
from os import path,listdir,remove
from Codificacion.metrics import euclideanDistance,hammingDistance
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score,silhouette_score,adjusted_rand_score
import numpy as np

cods=['dwt','gabor','pca','dct']
clsts=['k-means','b-means','k-means++','avg','ward','hdbscan']
labels=[]

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
	registered=listdir(path.join("Templates",code))
	for i in ids:
		ims=listdir(f"{dir}/{i}/")
		paths.append(f"{dir}/{i}/{ims[0]}")
		y_true.append(i)
		if ims[0][:-4]+'.mat' not in registered:
			register(f"{dir}/{i}/{ims[0]}",code,i)
	return paths,y_true

def getTemplates(cod:str):
	entrys=listdir(path.join("Templates",cod))
	
	temps=[]
	ids=[]
	
	for temp in entrys:
		template=loadmat(path.join("Templates",cod,temp))
		ids.append(''.join(template['id']))
		temps.append(template['template'].flatten())
	return temps,ids

def test(COD:str,CLST:str, n_clusters:int=3):
	dist=euclideanDistance
	if COD=='gabor':
		dist=hammingDistance
		#leer imagenes
	regs,y_true =registerDir("CASIA\\TEST",COD)
		#leer plantillas
	print("Leyendo plantillas...")
	
	temps,ids=getTemplates(COD)

	t_scores=[]	
	preds=[]
	#crear clusters
	print("Generando clusteres...")
	match CLST:
		case 'k-means':
			clusters=KMeans(n_clusters=n_clusters,random_state=0).fit(temps)
			labels.append((clusters.labels_,COD))
			table=joinTable(clusters,ids,temps)
			print("Evaluando...")
			for r in regs:
				id,im,dis,time,comps=searchCents(r,COD,clusters,table,dist)
				t_scores.append(time)
				preds.append(id)
			sil=silhouette_score(temps,clusters.labels_)

		case 'b-means':
			clusters=BisectingKMeans(n_clusters=n_clusters,random_state=0).fit(temps)
			labels.append((clusters.labels_,COD))
			table=joinTable(clusters,ids,temps)
			print("Evaluando...")
			for r in regs:
				id,im,dis,time,comps=searchCents(r,COD,clusters,table,dist)
				t_scores.append(time)
				preds.append(id)
			sil=silhouette_score(temps,clusters.labels_)

		case 'k-means++':
			clusters=KMeans(n_clusters=n_clusters,random_state=0,init='k-means++').fit(temps)
			labels.append((clusters.labels_,COD))
			table=joinTable(clusters,ids,temps)
			print("Evaluando...")
			for r in regs:
				id,im,dis,time,comps=searchCents(r,COD,clusters,table,dist)
				t_scores.append(time)
				preds.append(id)
			sil=silhouette_score(temps,clusters.labels_)
		
		case 'ward':
			tree=AgglomerativeClustering(n_clusters=n_clusters,linkage='ward').fit(temps)
			labels.append((tree.labels_,COD))
			table=joinTable(tree,ids,temps)
			print("Evaluando...")
			for r in regs:
				id,im,dis,time,comps=searchTree(r,COD,tree,table,euclideanDistance)
				t_scores.append(time)
				preds.append(id)
			sil=silhouette_score(temps,tree.labels_)

		case 'avg':
			tree=AgglomerativeClustering(n_clusters=n_clusters,linkage='average').fit(temps)
			labels.append((tree.labels_,COD))
			table=joinTable(tree,ids,temps)
			print("Evaluando...")
			for r in regs:
				id,im,dis,time,comps=searchTree(r,COD,tree,table,euclideanDistance)
				t_scores.append(time)
				preds.append(id)
			sil=silhouette_score(temps,tree.labels_)
			
		case 'hdbscan':
			db=HDBSCAN(min_cluster_size=3,max_cluster_size=35, store_centers='centroid',copy=False).fit(temps)
			labels.append((db.labels_,COD))
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
	"""with open("results.txt", "w") as f:
		f.write(f"Cluster_type\tn_clust\tCod\tAcc\tPres\tF1\tRec\tAvgT\tSil\n")
		f.close()"""
	
	for COD in cods:
		registered=listdir(path.join("Templates",COD))
		for r in registered: remove(path.join("Templates",COD,r))
		#for n in range(3,20):
		n=3
		for CLST in clsts:
			y_true,preds,sil,avg_t=test(COD,CLST,n)
			acc=accuracy_score(y_true,preds)
			pres=precision_score(y_true,preds,average='micro')
			f1=f1_score(y_true,preds,average='micro')
			rec=recall_score(y_true,preds,average='micro')
			print(f"Resultados de {COD} indexado con {CLST} {n}")
			print(f"Pres:{pres}	TR:{avg_t} SS:{sil}")
			f=open("results.txt", "a")
			f.write("{:}\t{:}\t{:}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(CLST,n,COD,acc,pres,f1,rec,avg_t,sil))
			f.close()
	f=open("ARI.txt",'w')
	f.write("L1\tL2\tCOD\tARI\n")
	for i in range(1,len(labels)):
		for j in range(i):
			if labels[i][1]==labels[j][1]:
				score=adjusted_rand_score(labels[i][0],labels[j][0])
				f.write("{:}\t{:}\t{:}\t{:.4f}\n".format(clsts[i%len(clsts)],clsts[j%len(clsts)],labels[i][1],score))
	f.close()

