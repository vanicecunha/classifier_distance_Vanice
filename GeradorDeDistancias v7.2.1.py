# -*- coding: utf-8 -*-
'''
	Essa versão parametriza a execução das distancias e aplica novas regras de aceite baseada
	em cada fluxo Pi x Qi
	
	Para rodar o script, execute:
		python GeradorDeDistancias v7.2.py

'''

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
CLASSE = {'bittorrent': 'P2P file-sharing', 'edonkey': 'P2P file-sharing', 'flashaudiolive': 'Live Streaming', 
'flashvideolive': 'Live Streaming', 'flashvideoondemand': 'Streaming ondemand', 'flashaudioondemand': 'Streaming ondemand',
'ftp': 'File Transfer',
'gamingrunscape': 'P2P Video' , 'gamingwaroflegends': 'P2P Video', 'gnutella': 'P2P file-sharing',
'httpaudioondemand': 'Streaming ondemand', 'httpdownload': 'HTTP Download','httpvideoondemand': 'Streaming ondemand',
'mmsaudiolive': 'Live Streaming', 'mmsvideolive': 'Live Streaming', 'ppstream': 'P2P Video', 'rtspaudiolive': 'Live Streaming',
'rtspaudioondemand': 'Streaming ondemand', 'rtspvideolive': 'Live Streaming','sftp': 'File Transfer','skype': 'Vo IP',
'sopcast': 'P2P Video', 'ssh': 'Remote Session','streaming1': 'Live Streaming','streaming2': 'Live Streaming',
'streaming3': 'Live Streaming','telnet': 'File Transfer','tvu': 'P2P Video','webbrowsing': 'Web Browsing' }

Ten_NoMatch = ['bittorrent',
'edonkey',
#'flashvideolive',
'flashvideoondemand',
'flashaudioondemand',
'ftp',
'gamingrunscape',
#'gamingwaroflegends',
'gnutella',
'httpdownload',
'httpvideoondemand',
'mmsaudiolive',
'mmsvideolive',
'rtspaudiolive',
#'rtspaudioondemand',
'sopcast',
'ssh',
#'streaming1',
#'streaming2',
#'streaming3',
#'telnet',
'tvu',
'webbrowsing'
]

TwentyFive = ['rtspvideolive',
'sftp',
'flashaudiolive'
]

One = ['ppstream'
]

DistMinima = ['httpaudioondemand',
'skype'
]

#Funções úteis
#Lê tracers e retorna lista de tuplas [ip, ip, tam]
def getDataFromFile(filename, header=False):
	fluxos = {}
	with open(filename, 'r') as f:
		spamreader = f
		if filename.endswith('.csv'):
			spamreader = csv.reader(f, delimiter=',')
		for line in spamreader:
			l = line.replace('\n','').split('***')
			fluxos[l[0]] = np.asarray(l[1].replace('[','').replace(']','').split(','), dtype=float)
			#MH[l[0]] = scipy.stats.hmean(fluxos[l[0]][fluxos[l[0]] > 0])
	return fluxos

def getDataFromFileIndividual(filename, header=False):
	freqRelativa = []
	with open(filename, 'r') as f:
		spamreader = f
		if filename.endswith('.csv'):
			spamreader = csv.reader(f, delimiter=',')
		for line in spamreader:
			freqRelativa = np.asarray(line.replace('[','').replace(']','').split(','),dtype=float)
	
	return freqRelativa, np.mean(freqRelativa), np.std(freqRelativa)
	
#Escreve uma lista em um arquivo
def writeFile(filename, lista):
	f = open(filename, 'w')
	for item in lista:
		f.write(str(item).replace('[','').replace(']','')+'\n')
	f.close()
	
#Escreve uma string em um arquivo já existente
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

def getDistanciaMahalanobis(x, y):
	i = 100
	k = j = 1
	xx = x.reshape(i,j*k).T
	yy = y.reshape(i,j*k).T
	e = xx-yy
	X = np.vstack([xx,yy])
	V = np.cov(X.T) 
	p = np.linalg.inv(V)
	D = np.sqrt(np.sum(np.dot(e,p) * e, axis = 1))
	return transformaIntervalo(D)
	
def getDistanciaQquadrado(lista1, lista2):
	#return abs(np.nansum(np.ma.masked_invalid(((abs(lista1 - lista2) -0.5)*(abs(lista1 - lista2) -0.5))/lista2)))
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

def getKeyFromValue(dictionary, search_value):
	for key, value in dictionary.iteritems():    # for name, age in dictionary.iteritems():  (for Python 2.x)
		if search_value[0] in value or search_value[1] in value:
			return True
	return False

def inList(args):
	global arquivoColetivo, protocoloAtual, PROTOCOLOS
	i = 0
	#Faz a validação com o ground truth (lista PROTOCOLOS), comparando se os ips de origem ou destino da tupla em questão realmente
	#pertencem ou não a classe da qual foi classificada
	for items in args:
		item = items[0].replace(']','').replace('[','').replace(' ','').replace("'",'').split(',')
		item1 = items[1].replace(']','').replace('[','').replace(' ','').replace("'",'').split(',')
		#if ((PROTOCOLOS[arquivoColetivo].get(CLASSE[protocoloAtual]) != None and (item[0] in PROTOCOLOS[arquivoColetivo][CLASSE[protocoloAtual]] or item[1] in PROTOCOLOS[arquivoColetivo][CLASSE[protocoloAtual]])) or (item[0] in item1 or item[1] in item1)):
		if (item[0] in item1 and item[1] in item1):
			i+=1
	return i
		
def classificaDistancias(nome, lista, tracersMisturados):
	global Qi, mediaConhecido, desvioConhecido, tracerIndividual, protocoloAtual, PROTOCOLOS, arquivoColetivo, Pi, Test
	
	#print 'Distância '+nome

	# Loop para verificar os aceitos de acordo com as distâncias das amostras Qi na lista.
	# Se alguma condição de aceite não for satisfeita, considero como rejeitado
	
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
	for r in minRange:
		aceitos = []
		rejeitados = []
		for i in range(len(lista)):
			dist = lista[i]
			#Se o valor da distância estiver no intervalo de aceite para cada fluxo Pi x Qi, aceito
			#item = Qi[i].replace(']','').replace('[','').replace(' ','').replace("'",'').split(',')
			#item1 = Pi[i].replace(']','').replace('[','').replace(' ','').replace("'",'').split(',')
			if dist <= minima and minima < r:
				#if not(item[0] in item1 and item[1] in item1):
				#	print dist, (item[0] in item1 and item[1] in item1), minima, mediana, media, dp
					#plt.plot(Test[i][0])
					#plt.plot(Test[i][1])
					#plt.show()
					
				aceitos.append([Qi[i], Pi[i]])
			#Senão é automaticamente rejeitado
			else:
				#if (item[0] in item1 and item[1] in item1):
				#	print dist, (item[0] in item1 and item[1] in item1), minima, mediana, media, dp
				rejeitados.append([Qi[i], Pi[i]])
		
		TP = 0
		TN = 0
		FP = 0
		FN = 0
		#print 'Aceitos(qtd)',len(aceitos)
		#print 'Rejeitados(qtd)',len(rejeitados)

		
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
	print 'Max threshold ', maxR
	return maxMatrix
		
def getMatrizConfusao(lista, nome, toFile = False):
	global nomeProtocolo
	
	TP = np.sum([pair[0] for pair in lista])
	TN = np.sum([pair[1] for pair in lista])
	FP = np.sum([pair[2] for pair in lista])
	FN = np.sum([pair[3] for pair in lista])
	
	print 'Matriz de confusão -',nome
	saida = 'Matriz de confusão -'+ nome+ '\n'
	
	print 'Protocolo/Aplicação -', nomeProtocolo
	saida = 'Protocolo/Aplicação -'+ nomeProtocolo+ '\n'
	
	print 'Classe -', CLASSE[nomeProtocolo]
	saida = 'Classe -'+CLASSE[nomeProtocolo]+ '\n'
	
	print 'TP=', TP,' FP=', FP
	saida += 'TP='+str(TP)+' FP='+str(FP)+'\n'
	
	print 'FN=', FN,' TN=', TN
	saida += 'FN='+str(FN)+' TN='+str(TN)+'\n'
		
	print 'Accurácia: ', (TP + TN)/(1.0*(TN+TP+FP+FN))
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
		f = open('TestesMin500_Protocolo/Resultados_FR/Matriz Confusão '+nome +'-'+nomeProtocolo+'.txt', 'w')
		f.write(saida)
		f.close()




nomeProtocolo = ""
protocoloAtual = ""
arquivoColetivo = ""
tracerIndividual = ""

if not os.path.exists('TestesMin500_Protocolo/Resultados_FR/'):
	os.makedirs('TestesMin500_Protocolo/Resultados_FR/')

	
#Prepara variáveis para matriz de confusão geral para cada distância por protocolo
matriz_eucl_global = []
matriz_leib_global = []
matriz_woot_global = []
matriz_hell_global = []
matriz_jens_global = []
matriz_bhat_global = []
matriz_kolm_global = []
matriz_qqua_global = []

#Médias e DP dos tracers
mediaConhecido = 0 
desvioConhecido = 0


individuaisFilenames = sorted(os.listdir('Individual_FR_new/'))
coletivosFilenames = sorted(os.listdir('Coletivo_FR_new/'))

#print 'Arquivos individuais encontrados:', individuaisFilenames, '\n'
#print 'Arquivos coletivos encontrados:', coletivosFilenames, '\n'

# ---------------------------------------------
# Parte2	 
# Lê tracers desconhecido
tracersMisturadosGeral = {}

	
for arquivoColetivo in coletivosFilenames:
	TRACER_MIXED = 'Coletivo_FR_new/'+arquivoColetivo
	#print 'Lendo tracer desconhecido '+ arquivoColetivo+ '...\n'
	
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

		if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Q-quadrado':	
			getMatrizConfusao(matriz_qqua_global, 'Q-quadrado',toFile=True)	
			
		matriz_eucl_global = []
		matriz_leib_global = []
		matriz_woot_global = []
		matriz_hell_global = []
		matriz_jens_global = []
		matriz_bhat_global = []
		matriz_kolm_global = []
		matriz_qqua_global = []
	
	
	print 'Aplicação/Protocolo Atual:', protocoloAtual, '\n'
	
	TRACER_INDIVIDUAL = 'Individual_FR_new/'+tracerIndividual
	tracersIndividuaisNew = getDataFromFile(TRACER_INDIVIDUAL, header=True) 
	
	
	#freqRelativa, mediaConhecido, desvioConhecido = getDataFromFileIndividual(TRACER_INDIVIDUAL, header=True)
	
	#print 'Média:', mediaConhecido, 'DP:',desvioConhecido,'\n'

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
			#print 'Arquivo Coletivo: ',arquivoColetivo, 'Qtd de fluxos:',len(tracersMisturadosGeral[arquivoColetivo])
			#print 'Total de amostras Qi:', len(tracersMisturadosGeral[arquivoColetivo])
			#print 'Processando amostras Qi...\n'
			#print '\t', arquivoColetivo
			#piQiDistDic[str(fluxoIndividual)+"_individual"][arquivoColetivo] = {}
			
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
				
				if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Q-quadrado':	
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
		
		if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Q-quadrado':
			dist = classificaDistancias('Q-quadrado', Qi_distancia_qqua,  tracersMisturadosGeral[arquivoColetivo])
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
			
	if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Q-quadrado':
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

if rodarScriptParaDist == 'Todos' or rodarScriptParaDist == 'Q-quadrado':	
	getMatrizConfusao(matriz_qqua_global, 'Q-quadrado',toFile=True)

print '\nTempo de execução:', time.time() - inicio, 'segundos.'
 
