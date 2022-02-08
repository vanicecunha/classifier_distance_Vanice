# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')
import numpy as np
import math
import random
import scipy.stats
import csv
import gc
import time
import os
import re
import sys
import json
from scipy.stats import entropy
import matplotlib.pyplot as plt
import hashlib
from sklearn.metrics import roc_curve, auc

#Automatic Garbage Collector
gc.enable()

inicio = time.time()


# Define as classes com base no IP de cada aplicação, para cada arquivo coletivo
PROTOCOLOS = {'labredes_05082011_1_CLEAN': {'Vo IP': ['10.0.2.144'], 
'P2P file-sharing': ['10.0.2.106', '10.0.2.153', '10.0.2.138', '10.0.2.198', '10.0.2.149', '10.0.2.125'], 
'Streaming ondemand': ['10.0.2.184', '10.0.2.186', '10.0.2.180'], 
'Live Streaming': ['10.0.2.148', '10.0.2.112', '10.0.2.118'], 
'HTTP Download': ['10.0.2.142'], 
'P2P Video': ['10.0.2.157', '10.0.2.162', '10.0.2.171'],
'File Transfer': ['10.0.2.178', '10.0.3.178', '10.0.3.108']},

'labredes_05082011_2_CLEAN': {'Vo IP': ['10.0.2.144'], 
'P2P file-sharing': ['10.0.2.138', '10.0.2.198', '10.0.2.149', '10.0.2.125', '10.0.2.106', '10.0.2.153'], 
'Streaming ondemand': ['10.0.2.112', '10.0.2.118', '10.0.2.148', '10.0.2.187'], 
'Live Streaming': ['10.0.2.103', '10.0.2.186', '10.0.2.184', '10.0.2.180'], 
'HTTP Download': ['10.0.2.142'], 
'P2P Video': ['10.0.2.171', '10.0.2.162', '10.0.2.157'],
'File Transfer': ['10.0.2.178', '10.0.3.178'],
'Remote Session': ['10.0.3.108','10.0.2.178']}, 

'labredes_05082011_3_CLEAN': {'Vo IP': ['10.0.2.144'], 
'P2P file-sharing': ['10.0.2.106', '10.0.2.153', '10.0.2.149', '10.0.2.125', '10.0.2.138', '10.0.2.198'], 
'Streaming ondemand': ['10.0.2.112', '10.0.2.187', '10.0.2.118', '10.0.2.148'], 
'Live Streaming': ['10.0.2.180', '10.0.2.103', '10.0.2.184', '10.0.2.186'], 
'Web Browsing': ['10.0.2.150'], 
'HTTP Download': ['10.0.2.142'], 
'P2P Video': ['10.0.2.171', '10.0.2.162', '10.0.2.157'],
'File Transfer': ['10.0.2.178', '10.0.3.108']}, 

'labredes_09082011_1_CLEAN': {'Vo IP': ['10.0.2.144'], 
'P2P file-sharing': ['10.0.2.138', '10.0.2.198', '10.0.2.149', '10.0.2.125', '10.0.2.106', '10.0.2.153'], 
'Streaming ondemand': ['10.0.2.180', '10.0.2.103', '10.0.2.184'], 
'Live Streaming': ['10.0.2.112', '10.0.2.148', '10.0.2.118', '10.0.2.187', '10.0.2.186'], 
'Web Browsing': ['10.0.2.182'], 
'HTTP Download': ['10.0.2.142'], 
'P2P Video': ['10.0.2.189', '10.0.2.157', '10.0.2.162', '10.0.2.171'],
'File Transfer': ['10.0.2.178', '10.0.3.178', '10.0.2.193', '10.0.3.108']},

'labredes_09082011_2_CLEAN': {'Vo IP': ['10.0.2.144'], 
'P2P file-sharing': ['10.0.2.138', '10.0.2.198', '10.0.2.106', '10.0.2.153', '10.0.2.149', '10.0.2.125'], 
'Streaming ondemand': ['10.0.2.184', '10.0.2.118', '10.0.2.112', '10.0.2.180'], 
'Live Streaming': ['10.0.2.148', '10.0.2.103', '10.0.2.187', '10.0.2.186'], 
'Web Browsing': ['10.0.2.182'], 
'HTTP Download': ['10.0.2.142'], 
'P2P Video': ['10.0.2.189', '10.0.2.171', '10.0.2.162', '10.0.2.157'],
'File Transfer': ['10.0.2.193', '10.0.3.108'],
'Remote Session': ['10.0.3.108','10.0.2.193']},

'labredes_09082011_3_CLEAN': {'Vo IP': ['10.0.2.144'], 
'P2P file-sharing': ['10.0.2.138', '10.0.2.198', '10.0.2.149', '10.0.2.125', '10.0.2.106', '10.0.2.153'], 
'Streaming ondemand': ['10.0.2.118', '10.0.2.184', '10.0.2.112', '10.0.2.180'], 
'Live Streaming': ['10.0.2.186', '10.0.2.148', '10.0.2.103', '10.0.2.187'], 
'Web Browsing': ['10.0.2.182'], 
'HTTP Download': ['10.0.2.142'], 
'P2P Video': ['10.0.2.171', '10.0.2.189', '10.0.2.162', '10.0.2.157'],
'File Transfer': ['10.0.2.193', '10.0.3.108'],
'Remote Session': ['10.0.3.108','10.0.2.193']},

'labredes_10082011_1_CLEAN': {'Vo IP': ['10.0.2.144'], 
'P2P file-sharing': ['10.0.2.138', '10.0.2.198', '10.0.2.106', '10.0.2.153', '10.0.2.149', '10.0.2.125'], 
'Streaming ondemand': ['10.0.2.180', '10.0.2.184', '10.0.2.103'], 
'Live Streaming': ['10.0.2.187', '10.0.2.186', '10.0.2.112', '10.0.2.148', '10.0.2.118'], 
'Web Browsing': ['10.0.2.182'], 
'HTTP Download': ['10.0.2.142'], 
'P2P Video': ['10.0.2.162', '10.0.2.171', '10.0.2.189', '10.0.2.181', '10.0.2.157'],
'File Transfer': ['10.0.2.193', '10.0.3.108'],
'Remote Session': ['10.0.3.108','10.0.2.193']},  

'labredes_10082011_2_CLEAN': {'Vo IP': ['10.0.2.144'], 
'P2P file-sharing': ['10.0.2.106', '10.0.2.153', '10.0.2.149', '10.0.2.125', '10.0.2.138', '10.0.2.198'], 
'Streaming ondemand': ['10.0.2.187', '10.0.2.184'], 
'Live Streaming': ['10.0.2.103', '10.0.2.118', '10.0.2.112', '10.0.2.186', '10.0.2.148', '10.0.2.180'], 
'Web Browsing': ['10.0.2.182'], 
'HTTP Download': ['10.0.2.142'], 
'P2P Video': ['10.0.2.157', '10.0.2.162', '10.0.2.189', '10.0.2.171', '10.0.2.181'],
'File Transfer': ['10.0.2.193', '10.0.3.108']}, 

'labredes_10082011_3_CLEAN': {'Vo IP': ['10.0.2.144'], 
'P2P file-sharing': ['10.0.2.138', '10.0.2.198', '10.0.2.149', '10.0.2.125', '10.0.2.106', '10.0.2.153'], 
'Streaming ondemand': ['10.0.2.187', '10.0.2.184'], 
'Live Streaming': ['10.0.2.180', '10.0.2.186', '10.0.2.118', '10.0.2.112', '10.0.2.103', '10.0.2.148'], 
'Web Browsing': ['10.0.2.182'], 
'HTTP Download': ['10.0.2.142'], 
'P2P Video': ['10.0.2.157', '10.0.2.171', '10.0.2.181', '10.0.2.162', '10.0.2.189'],
'File Transfer': ['10.0.2.193', '10.0.3.108']}}
 
# Mapeamento de protocolo para classe
CLASSE = {  'bittorrent': 'P2P file-sharing', 
            'edonkey': 'P2P file-sharing', 
            'flashaudiolive': 'Live Streaming', 
            'flashvideolive': 'Live Streaming',
            'flashvideoondemand': 'Streaming ondemand',
            'flashaudioondemand': 'Streaming ondemand',
            'ftp': 'File Transfer',
            'gamingrunscape': 'Game' , 
            'gamingwaroflegends': 'Game', 
            'gnutella': 'P2P file-sharing',
            'httpaudioondemand': 'Streaming ondemand', 
            'httpdownload': 'HTTP Download',
            'httpvideoondemand': 'Streaming ondemand',
            'mmsaudiolive': 'Live Streaming', 
            'mmsvideolive': 'Live Streaming', 
            'ppstream': 'P2P Video', 
            'rtspaudiolive': 'Live Streaming',
            'rtspaudioondemand': 'Streaming ondemand', 
            'rtspvideolive': 'Live Streaming',
            'sftp': 'File Transfer',
            'skype': 'Vo IP',
            'sopcast': 'P2P Video', 
            'ssh': 'Remote Session',
            'streaming1': 'Live Streaming',
            'streaming2': 'Live Streaming',
            'streaming3': 'Live Streaming',
            'telnet': 'File Transfer',
            'tvu': 'P2P Video',
            'webbrowsing': 'Web Browsing' }


IgnoreList = ['flashvideolive', 'ssh', 'rtspaudioondemand', 'streaming1', 'streaming2', 'streaming3', 
'gamingwaroflegends', 'telnet']

#Funções úteis
#Lê tracers e retorna lista de fluxos para cada arquivo coletivo/individual
def getDataFromFile(filename, header=False):
	fluxos = {}
	with open(filename, 'r') as f:
		spamreader = f
		if filename.endswith('.csv'):
			spamreader = csv.reader(f, delimiter=',')
		for line in spamreader:
			l = line.replace('\n','').split('***')
			fluxos[l[0]] = np.asarray(l[1].replace('[','').replace(']','').split(','), dtype=float)
	return fluxos
	
#Escreve uma lista em um arquivo - Não utilizada nessa versão
def writeFile(filename, lista):
	f = open(filename, 'w')
	for item in lista:
		f.write(str(item).replace('[','').replace(']','')+'\n')
	f.close()
	
#Escreve uma string em um arquivo já existente - Não utilizada nessa versão
def appendToFile(filename, tupla):
	f = open(filename, 'a+')
	f.write(tupla.replace('[','').replace(']','').replace(',','')+'\n')
	f.close()
	
def getFreqAcumulada(lista):
	return np.cumsum(lista)	
		
def getDistanciaEuclidiana(lista1, lista2):
	dist = abs(np.linalg.norm(lista1 - lista2))	
	return transformaIntervalo(dist)
	
def getDistanciaKullbackLeibler(lista1, lista2):
	dist = scipy.stats.entropy(lista1+1e-8, lista2+1e-8)
	return transformaIntervalo(dist)	

def getDistanciaWootters(lista1, lista2):
	dist = np.arccos(transformaIntervalo(np.sum(np.sqrt(lista1 * lista2))))
	return transformaIntervalo(dist)
	
def getDistanciaHellinger(lista1, lista2):
	dist = np.sqrt(0.5*np.sum((np.sqrt(lista1)-np.sqrt(lista2))**2))
	return transformaIntervalo(dist)
	
def getDistanciaJensen(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
   # normalize
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    dist =  (entropy(p, m) + entropy(q, m)) / 2	
    return transformaIntervalo(dist)
	
def getDistanciaBhattacharyya(lista1, lista2):
	dist =  -1*np.log(np.sum(np.sqrt(lista1 * lista2))+1e-8)
	return transformaIntervalo(dist)
	
def getDistanciaKolmogorovSmirnov(lista1, lista2):
	d,_ = scipy.stats.ks_2samp(lista1,lista2)
	return transformaIntervalo(d)
	
def getDistanciaQquadrado(lista1, lista2):
	dist = 0.5*np.sum((lista1-lista2)**2/(lista1+lista2+1e-8))
	return transformaIntervalo(dist)

#Transforma em intervalo [0, 1] caso valor não faça parte do mesmo 
def transformaIntervalo(n):	
	aux = float(format(n, 'f'))
	if abs(aux) == 0:
		return 0
	if aux == 1:
		return 1
	if n >=0 and n <= 1:
		return n
					
	return abs(n) / (1 + abs(n))

#Faz a validação ground truth, comparando se os ips de origem e destino do fluxo coletivo em questão realmente
#pertencem ou não a classe individual da qual foi classificada
def inList(args):
	global arquivoColetivo, protocoloAtual
	i = 0
	for items in args:
		item = items[0].replace(']','').replace('[','').replace(' ','').replace("'",'').split(',')
		item1 = items[1].replace(']','').replace('[','').replace(' ','').replace("'",'').split(',')
		if (item[0] in item1 and item[1] in item1):
			i+=1
	return i
		
def classificaDistancias(nome, lista, tracersMisturados):
	global Qi, tracerIndividual, protocoloAtual, arquivoColetivo, Pi, ROC, Test
		
	#print 'Distância '+nome

	# Loop para verificar os aceitos de acordo com as distâncias das amostras Qi na lista.
	# Se a condição de aceite não for satisfeita, considero como rejeitado
	
	#maxima = np.max(lista)
	minima = np.min(lista)
	#media = np.mean(lista)
	#mediana = np.median(lista)
	#dp = np.std(lista)
	#lista = np.array(lista)
	#mh = scipy.stats.hmean(lista[lista > 0])
	#print minima, maxima, media, mediana, dp
	minRange = [0.0001, 0.001, 0.01, 0.025, 1]
	maxFmeasure = -1
	maxR = 0
	maxMatrix = []
	maxROC_predito = []
	ROC_esperado = []
	for r in minRange:
		aceitos = []
		rejeitados = []
		ROC_predito = []
		for i in range(len(lista)):
			dist = lista[i]
			#Se o valor da distância for igual a minima, e a minima for menor que minRange atual, aceito
			
			#só preciso rodar essa parte 1 vez, pra verificar as classes esperadas
			if maxFmeasure == -1:
				#Para este fluxo, salva 1 caso fluxo protocolo atual = fluxo protocolo coletivo ou 0 c.c. (valor real)
				item = Qi[i].replace(']','').replace('[','').replace(' ','').replace("'",'').split(',')
				item1 = Pi[i].replace(']','').replace('[','').replace(' ','').replace("'",'').split(',')
				if item[0] in item1 and item[1] in item1:
					ROC_esperado.append(1)
				else: 
					ROC_esperado.append(0)
			
			#Para este fluxo, salva 1 caso criterio seja satisfeito, 0 c.c. (valor predito)
			if dist == minima and minima < r:
				#if not(item[0] in item1 and item[1] in item1):
				#	print dist, (item[0] in item1 and item[1] in item1), minima, mediana, media, dp
					#plt.plot(Test[i][0])
					#plt.plot(Test[i][1])
					#plt.show()
				ROC_predito.append(1)
				aceitos.append([Qi[i], Pi[i]])
				
			#Senão é automaticamente rejeitado
			else:
				#if (item[0] in item1 and item[1] in item1):
				#	print dist, (item[0] in item1 and item[1] in item1), minima, mediana, media, dp
				ROC_predito.append(0)
				rejeitados.append([Qi[i], Pi[i]])
				
		
		TP = 0
		TN = 0
		FP = 0
		FN = 0

		
		#print 'Checando lista de aceitos'
		results = inList(aceitos)
		TP = results
		FP = len(aceitos) - results
		
		#print 'Checando lista de rejeitados'		
		results2 = inList(rejeitados)
		TN = len(rejeitados) - results2
		FN = results2 
		
		precision = 0.0
		if (1.0*FP + TP) > 0 :
			precision = TP/(1.0*FP + TP) 
		
		recall = 0.0
		if (1.0*TP + FN) > 0:
			recall = TP/(1.0*TP + FN)  
		
		denominador = precision + recall if precision + recall != 0 else 1
		fmeasure = (2*precision*recall)/denominador
		if fmeasure > maxFmeasure:
			maxFmeasure = fmeasure
			maxMatrix = [TP, TN, FP, FN]
			maxR = r
			maxROC_predito = ROC_predito
			
	if nome not in ROC:
		ROC[nome] = [[], []]

	#concatena as classes esperadas e preditas para cada distancia
	ROC[nome][0] = ROC[nome][0] +  	ROC_esperado	
	ROC[nome][1] = ROC[nome][1] +  	maxROC_predito
	
	#print 'Max threshold ', maxR
	return maxMatrix

#Devolve a matrix de confusão de todas as matrizes de confusao de terminada classe		
def getMatrizConfusao(lista, nome, toFile = False):
	global nomeProtocolo
	
	TP = np.sum([pair[0] for pair in lista])
	TN = np.sum([pair[1] for pair in lista])
	FP = np.sum([pair[2] for pair in lista])
	FN = np.sum([pair[3] for pair in lista])
	
	print'Matriz de confusão -',nome
	saida = 'Matriz de confusão -'+ nome+ '\n'
	
	print'Protocolo/Aplicação -', nomeProtocolo
	saida = 'Protocolo/Aplicação -'+ nomeProtocolo+ '\n'
	
	print'Classe -', CLASSE[nomeProtocolo]
	saida = 'Classe -'+CLASSE[nomeProtocolo]+ '\n'
	
	print'TP=', TP,' FP=', FP
	saida += 'TP='+str(TP)+' FP='+str(FP)+'\n'
	
	print'FN=', FN,' TN=', TN
	saida += 'FN='+str(FN)+' TN='+str(TN)+'\n'
		
	print'Accurácia: ', (TP + TN)/(1.0*(TN+TP+FP+FN))
	saida += 'Accurácia: '+str((TP + TN)/(1.0*(TN+TP+FP+FN)))+'\n'
	
	precision = 0.0
	if (1.0*FP + TP) > 0 :
		precision = TP/(1.0*FP + TP) 
	print 'Precisão: ',precision
	saida += 'Precisão: '+str(precision)+'\n'
	
	recall = 0.0
	if (1.0*TP + FN) > 0:
		recall = TP/(1.0*TP + FN)  
	print 'Recall: ', recall
	saida += 'Recall: '+str(recall)+'\n'
	
	denominador = precision + recall if precision + recall != 0 else 1
	fmeasure = (2*precision*recall)/denominador
	print 'F-measure: ', fmeasure, '\n'
	saida += 'F-measure: '+str(fmeasure)+'\n'
		
	if toFile:
		f = open('Resultadosv7.2_BaseCompleta/Resultados_FR/Matriz Confusão '+nome +'-'+nomeProtocolo+'.txt', 'w')
		f.write(saida)
		f.close()


def writeROCCurves():
	global ROC, nomeProtocolo, rodarScriptParaDist
	# Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
 	n_classes = []
	if (rodarScriptParaDist == 'Todos'):
		n_classes = ['Euclidiana', 'Kullback-Leibler', 'Wootters', 'Hellinger', 'Jensen', 'Bhattacharyya', 'Kolmogorov-Smirnov', 'Chi-square']
	else:
		n_classes = [rodarScriptParaDist]
 	for i in range(len(n_classes)):
		#print len(ROC[n_classes[i]]), len(ROC[n_classes[i]][0]), len(ROC[n_classes[i]][1])
		fpr[i], tpr[i], _ = roc_curve(ROC[n_classes[i]][0], ROC[n_classes[i]][1])
		roc_auc[i] = auc(fpr[i], tpr[i])

	# Plot all ROC curves
	plt.figure()
	lw = 1
	for i in range(len(n_classes)):
		plt.plot(fpr[i], tpr[i], lw=lw,
				 label='{0} (area = {1:0.2f})'
				 ''.format(n_classes[i].replace('Euclidiana', 'Euclidean'), roc_auc[i]))

	plt.plot([0, 1], [0, 1], 'k--', lw=lw)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve: '+nomeProtocolo)
	plt.legend(loc="lower right")
	#plt.show()
	plt.savefig('Resultadosv7.2_BaseCompleta/ROC_plots/'+nomeProtocolo+'_roc.png')
	ROC = {}

#-------------------- Main ---------------------
nomeProtocolo = ""
protocoloAtual = ""
arquivoColetivo = ""
tracerIndividual = ""

if not os.path.exists('Resultadosv7.2_BaseCompleta/Resultados_FR/'):
	os.makedirs('Resultadosv7.2_BaseCompleta/Resultados_FR/')

if not os.path.exists('Resultadosv7.2_BaseCompleta/ROC_plots/'):
	os.makedirs('Resultadosv7.2_BaseCompleta/ROC_plots/')
	
#Prepara variáveis para matriz de confusão geral para cada distância por protocolo
matriz_eucl_global = []
matriz_leib_global = []
matriz_woot_global = []
matriz_hell_global = []
matriz_jens_global = []
matriz_bhat_global = []
matriz_kolm_global = []
matriz_qqua_global = []

ROC = {}


individuaisFilenames = sorted(os.listdir('Individual_FR_LessSamples/'))
coletivosFilenames = sorted(os.listdir('Coletivo_FR_MoreSamples/'))

#print 'Arquivos individuais encontrados:', individuaisFilenames, '\n'
#print 'Arquivos coletivos encontrados:', coletivosFilenames, '\n'

# ---------------------------------------------
# Parte2	 
# Lê tracers desconhecido
tracersMisturadosGeral = {}

	
for arquivoColetivo in coletivosFilenames:
	TRACER_MIXED = 'Coletivo_FR_MoreSamples/'+arquivoColetivo
	print 'Lendo tracer desconhecido '+ arquivoColetivo+ '...\n'
	
	tracersMisturados = getDataFromFile(TRACER_MIXED, header=True) 
	
	tracersMisturadosGeral[arquivoColetivo.replace('FR_','')] = tracersMisturados


#Roda o script para 1 distancia especifica
rodarScriptParaDist = 'Todos'

if len(sys.argv) > 1:
	rodarScriptParaDist = str(sys.argv[1])

# --------------------------------------------
for index in range(0, len(individuaisFilenames)):
	# Parte 1
	# Processamento dos tracers individuais
	tracerIndividual = individuaisFilenames[index]
	#print tracerIndividual
	protocoloAtual = re.sub(r'(_labredes)?_\d+', '', tracerIndividual).replace('FR_','')
	protocoloAtual = re.sub(r'^[^_]+_', '', protocoloAtual).replace('_','')
	
	if True:
 	#if protocoloAtual in ['ssh']:
		if nomeProtocolo != "" and protocoloAtual != nomeProtocolo:
			if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Euclidiana':
				getMatrizConfusao(matriz_eucl_global, 'Euclidiana', toFile=True)
				
			if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Kullback-Leibler':	
				getMatrizConfusao(matriz_leib_global, 'Kullback-Leibler',toFile=True)
				
			if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Wootters':	
				getMatrizConfusao(matriz_woot_global, 'Wootters',toFile=True)

			if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Hellinger':	
				getMatrizConfusao(matriz_hell_global, 'Hellinger',toFile=True)

			if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Jensen':	
				getMatrizConfusao(matriz_jens_global, 'Jensen',toFile=True)

			if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Bhattacharyya':	
				getMatrizConfusao(matriz_bhat_global, 'Bhattacharyya',toFile=True)	

			if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Kolmogorov-Smirnov':	
				getMatrizConfusao(matriz_kolm_global, 'Kolmogorov-Smirnov',toFile=True)	

			if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Chi-square':	
				getMatrizConfusao(matriz_qqua_global, 'Chi-square',toFile=True)	
				
			writeROCCurves()
				
			matriz_eucl_global = []
			matriz_leib_global = []
			matriz_woot_global = []
			matriz_hell_global = []
			matriz_jens_global = []
			matriz_bhat_global = []
			matriz_kolm_global = []
			matriz_qqua_global = []
		
		
		print 'Aplicação/Protocolo Atual:', protocoloAtual, '\n'
		
		TRACER_INDIVIDUAL = 'Individual_FR_LessSamples/'+tracerIndividual
		tracersIndividuaisNew = getDataFromFile(TRACER_INDIVIDUAL, header=True) 
		
		
		matriz_eucl = []
		matriz_leib = []
		matriz_woot = []
		matriz_hell = []
		matriz_jens = []
		matriz_bhat = []
		matriz_kolm = []
		matriz_qqua = []
		#print 'Arquivo Individual: ',tracerIndividual, 'Qtd de fluxos:',len(tracersIndividuaisNew)
		for fluxoIndividual in tracersIndividuaisNew:	
			freqRelativa = tracersIndividuaisNew[fluxoIndividual]
			
			for arquivoColetivo in tracersMisturadosGeral:
				
				Qi_distancia_euclidiana = []
				Qi_distancia_leiber = []
				Qi_distancia_wooters = []
				Qi_distancia_hellinger = []
				Qi_distancia_jensen = []
				Qi_distancia_bhattacharyya = []
				Qi_distancia_kolm = []
				Qi_distancia_qqua = []

				Qi = []
				Pi = []
				Test = []
				for fluxo in tracersMisturadosGeral[arquivoColetivo]:			
					Qi_freqRelativa = tracersMisturadosGeral[arquivoColetivo][fluxo]
					
					#Calcula distâncias 
					#Freq Relativa
					if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Euclidiana':
						distEucl = getDistanciaEuclidiana(freqRelativa, Qi_freqRelativa)
						Qi_distancia_euclidiana.append(distEucl)
					
					if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Kullback-Leibler':	
						distLeib = getDistanciaKullbackLeibler(freqRelativa, Qi_freqRelativa)
						Qi_distancia_leiber.append(distLeib)
					
					if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Wootters':	
						distWoott = getDistanciaWootters(freqRelativa, Qi_freqRelativa)
						Qi_distancia_wooters.append(distWoott)
					
					if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Hellinger':	
						distHell = getDistanciaHellinger(freqRelativa, Qi_freqRelativa)
						Qi_distancia_hellinger.append(distHell)
					
					if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Jensen':	
						distJens = getDistanciaJensen(freqRelativa, Qi_freqRelativa)
						Qi_distancia_jensen.append(distJens)
					
					if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Bhattacharyya':	
						distBhat = getDistanciaBhattacharyya(freqRelativa, Qi_freqRelativa)
						Qi_distancia_bhattacharyya.append(distBhat)
					
					if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Kolmogorov-Smirnov':	
						#Kolmogorov sempre usa freq acumulada
						distKolm = getDistanciaKolmogorovSmirnov(getFreqAcumulada(freqRelativa), getFreqAcumulada(Qi_freqRelativa))
						Qi_distancia_kolm.append(distKolm)		
					
					if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Chi-square':	
						distQqua = getDistanciaQquadrado(freqRelativa, Qi_freqRelativa)
						Qi_distancia_qqua.append(distQqua)
							
						
					Qi.append(fluxo)
					Pi.append(fluxoIndividual)
					Test.append([freqRelativa, Qi_freqRelativa])


			if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Euclidiana':
				dist = classificaDistancias('Euclidiana', Qi_distancia_euclidiana,  tracersMisturadosGeral[arquivoColetivo])
				matriz_eucl.append(dist)
			
			if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Kullback-Leibler':
				dist = classificaDistancias('Kullback-Leibler', Qi_distancia_leiber,  tracersMisturadosGeral[arquivoColetivo])
				matriz_leib.append(dist)
			
			if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Wootters':
				dist = classificaDistancias('Wootters', Qi_distancia_wooters,tracersMisturadosGeral[arquivoColetivo])
				matriz_woot.append(dist)
			
			if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Hellinger':
				dist = classificaDistancias('Hellinger', Qi_distancia_hellinger, tracersMisturadosGeral[arquivoColetivo])
				matriz_hell.append(dist)
			
			if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Jensen':
				dist = classificaDistancias('Jensen', Qi_distancia_jensen, tracersMisturadosGeral[arquivoColetivo])
				matriz_jens.append(dist)
			
			if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Bhattacharyya':
				dist = classificaDistancias('Bhattacharyya', Qi_distancia_bhattacharyya, tracersMisturadosGeral[arquivoColetivo])
				matriz_bhat.append(dist)
			
			if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Kolmogorov-Smirnov':
				dist = classificaDistancias('Kolmogorov-Smirnov', Qi_distancia_kolm, tracersMisturadosGeral[arquivoColetivo])
				matriz_kolm.append(dist)
			
			if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Chi-square':
				dist = classificaDistancias('Chi-square', Qi_distancia_qqua,  tracersMisturadosGeral[arquivoColetivo])
				matriz_qqua.append(dist)

		if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Euclidiana':
			for i in range(len(matriz_eucl)):
				matriz_eucl_global.append(matriz_eucl[i])
						
		if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Kullback-Leibler':
			for i in range(len(matriz_leib)):	
				matriz_leib_global.append(matriz_leib[i])
				
		if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Wootters':
			for i in range(len(matriz_woot)):	
				matriz_woot_global.append(matriz_woot[i])
				
		if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Hellinger':
			for i in range(len(matriz_hell)):	
				matriz_hell_global.append(matriz_hell[i])
				
		if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Jensen':
			for i in range(len(matriz_jens)):	
				matriz_jens_global.append(matriz_jens[i])
				
		if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Bhattacharyya':
			for i in range(len(matriz_bhat)):	
				matriz_bhat_global.append(matriz_bhat[i])
				
		if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Kolmogorov-Smirnov':
			for i in range(len(matriz_kolm)):	
				matriz_kolm_global.append(matriz_kolm[i])	
				
		if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Chi-square':
			for i in range(len(matriz_qqua)):	
				matriz_qqua_global.append(matriz_qqua[i])
		
		
		nomeProtocolo = protocoloAtual

	
	
if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Euclidiana':
	getMatrizConfusao(matriz_eucl_global, 'Euclidiana', toFile=True)
	
if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Kullback-Leibler':	
	getMatrizConfusao(matriz_leib_global, 'Kullback-Leibler',toFile=True)
	
if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Wootters':	
	getMatrizConfusao(matriz_woot_global, 'Wootters',toFile=True)

if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Hellinger':	
	getMatrizConfusao(matriz_hell_global, 'Hellinger',toFile=True)

if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Jensen':	
	getMatrizConfusao(matriz_jens_global, 'Jensen',toFile=True)

if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Bhattacharyya':	
	getMatrizConfusao(matriz_bhat_global, 'Bhattacharyya',toFile=True)	

if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Kolmogorov-Smirnov':	
	getMatrizConfusao(matriz_kolm_global, 'Kolmogorov-Smirnov',toFile=True)	

if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Chi-square':	
	getMatrizConfusao(matriz_qqua_global, 'Chi-square',toFile=True)

writeROCCurves()

print '\nTempo de execução:', time.time() - inicio, 'segundos.'
 
