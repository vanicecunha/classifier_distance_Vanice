# -*- coding: utf-8 -*-
'''
	Gera graficos de FR

'''
#Define limites para aceitar e para comparar médias e desvio padrão. 
#Intervalo [LIMITE_INFERIOR_ACEITA, LIMITE_SUPERIOR_ACEITA] é automaticamente aceito
#No intervalo (LIMITE_SUPERIOR_ACEITA, LIMITE_SUPERIOR_COMPARA] compara-se as médias e desvios padrões
#Intervalo (LIMITE_SUPERIOR_COMPARA, +infinito) é automaticamente rejeitado
LIMITE_INFERIOR_ACEITA = 0
LIMITE_SUPERIOR_ACEITA = 0.10
LIMITE_SUPERIOR_COMPARA = 0.59

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
import matplotlib.pyplot as plt


#Automatic Garbage Collector
gc.enable()

inicio = time.time()


# Define as classes com base no IP de cada aplicação, para cada arquivo coletivo
PROTOCOLOS = {'labredes_05082011_2_CLEAN': {'Vo IP': ['10.0.2.144'], 
'P2P file-sharing': ['10.0.2.138', '10.0.2.198', '10.0.2.149', '10.0.2.125', '10.0.2.106', '10.0.2.153'], 
'Streaming ondemand': ['10.0.2.112', '10.0.2.118', '10.0.2.148', '10.0.2.187'], 
'Live Streaming': ['10.0.2.103', '10.0.2.186', '10.0.2.184', '10.0.2.180'], 
'HTTP Download': ['10.0.2.142'], 
'P2P Video': ['10.0.2.171', '10.0.2.162', '10.0.2.157']}, 

'labredes_10082011_1_CLEAN': {'Vo IP': ['10.0.2.144'], 
'P2P file-sharing': ['10.0.2.138', '10.0.2.198', '10.0.2.106', '10.0.2.153', '10.0.2.149', '10.0.2.125'], 
'Streaming ondemand': ['10.0.2.180', '10.0.2.184', '10.0.2.103'], 
'Live Streaming': ['10.0.2.187', '10.0.2.186', '10.0.2.112', '10.0.2.148', '10.0.2.118'], 
'Web Browsing': ['10.0.2.182'], 
'HTTP Download': ['10.0.2.142'], 
'P2P Video': ['10.0.2.162', '10.0.2.171', '10.0.2.189', '10.0.2.181', '10.0.2.157']}, 

'labredes_09082011_3_CLEAN': {'Vo IP': ['10.0.2.144'], 
'P2P file-sharing': ['10.0.2.138', '10.0.2.198', '10.0.2.149', '10.0.2.125', '10.0.2.106', '10.0.2.153'], 
'Streaming ondemand': ['10.0.2.118', '10.0.2.184', '10.0.2.112', '10.0.2.180'], 
'Live Streaming': ['10.0.2.186', '10.0.2.148', '10.0.2.103', '10.0.2.187'], 
'Web Browsing': ['10.0.2.182'], 
'HTTP Download': ['10.0.2.142'], 
'P2P Video': ['10.0.2.171', '10.0.2.189', '10.0.2.162', '10.0.2.157']}, 

'labredes_10082011_2_CLEAN': {'Vo IP': ['10.0.2.144'], 
'P2P file-sharing': ['10.0.2.106', '10.0.2.153', '10.0.2.149', '10.0.2.125', '10.0.2.138', '10.0.2.198'], 
'Streaming ondemand': ['10.0.2.187', '10.0.2.184'], 
'Live Streaming': ['10.0.2.103', '10.0.2.118', '10.0.2.112', '10.0.2.186', '10.0.2.148', '10.0.2.180'], 
'Web Browsing': ['10.0.2.182'], 
'HTTP Download': ['10.0.2.142'], 
'P2P Video': ['10.0.2.157', '10.0.2.162', '10.0.2.189', '10.0.2.171', '10.0.2.181']}, 

'labredes_09082011_2_CLEAN': {'Vo IP': ['10.0.2.144'], 
'P2P file-sharing': ['10.0.2.138', '10.0.2.198', '10.0.2.106', '10.0.2.153', '10.0.2.149', '10.0.2.125'], 
'Streaming ondemand': ['10.0.2.184', '10.0.2.118', '10.0.2.112', '10.0.2.180'], 
'Live Streaming': ['10.0.2.148', '10.0.2.103', '10.0.2.187', '10.0.2.186'], 
'Web Browsing': ['10.0.2.182'], 
'HTTP Download': ['10.0.2.142'], 
'P2P Video': ['10.0.2.189', '10.0.2.171', '10.0.2.162', '10.0.2.157']}, 

'labredes_05082011_3_CLEAN': {'Vo IP': ['10.0.2.144'], 
'P2P file-sharing': ['10.0.2.106', '10.0.2.153', '10.0.2.149', '10.0.2.125', '10.0.2.138', '10.0.2.198'], 
'Streaming ondemand': ['10.0.2.112', '10.0.2.187', '10.0.2.118', '10.0.2.148'], 
'Live Streaming': ['10.0.2.180', '10.0.2.103', '10.0.2.184', '10.0.2.186'], 
'Web Browsing': ['10.0.2.150'], 
'HTTP Download': ['10.0.2.142'], 
'P2P Video': ['10.0.2.171', '10.0.2.162', '10.0.2.157']}, 

'labredes_10082011_3_CLEAN': {'Vo IP': ['10.0.2.144'], 
'P2P file-sharing': ['10.0.2.138', '10.0.2.198', '10.0.2.149', '10.0.2.125', '10.0.2.106', '10.0.2.153'], 
'Streaming ondemand': ['10.0.2.187', '10.0.2.184'], 
'Live Streaming': ['10.0.2.180', '10.0.2.186', '10.0.2.118', '10.0.2.112', '10.0.2.103', '10.0.2.148'], 
'Web Browsing': ['10.0.2.182'], 
'HTTP Download': ['10.0.2.142'], 
'P2P Video': ['10.0.2.157', '10.0.2.171', '10.0.2.181', '10.0.2.162', '10.0.2.189']}, 

'labredes_05082011_1_CLEAN': {'Vo IP': ['10.0.2.144'], 
'P2P file-sharing': ['10.0.2.106', '10.0.2.153', '10.0.2.138', '10.0.2.198', '10.0.2.149', '10.0.2.125'], 
'Streaming ondemand': ['10.0.2.184', '10.0.2.186', '10.0.2.180'], 
'Live Streaming': ['10.0.2.148', '10.0.2.112', '10.0.2.118'], 
'HTTP Download': ['10.0.2.142'], 
'P2P Video': ['10.0.2.157', '10.0.2.162', '10.0.2.171']}, 

'labredes_09082011_1_CLEAN': {'Vo IP': ['10.0.2.144'], 
'P2P file-sharing': ['10.0.2.138', '10.0.2.198', '10.0.2.149', '10.0.2.125', '10.0.2.106', '10.0.2.153'], 
'Streaming ondemand': ['10.0.2.180', '10.0.2.103', '10.0.2.184'], 
'Live Streaming': ['10.0.2.112', '10.0.2.148', '10.0.2.118', '10.0.2.187', '10.0.2.186'], 
'Web Browsing': ['10.0.2.182'], 
'HTTP Download': ['10.0.2.142'], 
'P2P Video': ['10.0.2.189', '10.0.2.157', '10.0.2.162', '10.0.2.171']}}
 
# Mapeamento de protocolo para classe
'''CLASSE = {'bittorrent': 'P2P file-sharing', 'edonkey': 'P2P file-sharing', 'flashaudiolive': 'Live Streaming', 
'flashvideolive': 'Live Streaming', 'flashvideoondemand': 'Streaming ondemand', 'flashaudioondemand': 'Streaming ondemand',
'ftp': 'File Transfer',
'gamingrunscape': 'P2P Video' , 'gamingwaroflegends': 'P2P Video', 'gnutella': 'P2P file-sharing',
'httpaudioondemand': 'Streaming ondemand', 'httpdownload': 'HTTP Download','httpvideoondemand': 'Streaming ondemand',
'mmsaudiolive': 'Live Streaming', 'mmsvideolive': 'Live Streaming', 'ppstream': 'P2P Video', 'rtspaudiolive': 'Live Streaming',
'rtspaudioondemand': 'Streaming ondemand', 'rtspvideolive': 'Live Streaming','sftp': 'File Transfer','skype': 'Vo IP',
'sopcast': 'P2P Video', 'ssh': 'Remote Session','streaming1': 'Live Streaming','streaming2': 'Live Streaming',
'streaming3': 'Live Streaming','telnet': 'File Transfer','tvu': 'P2P Video','webbrowsing': 'Web Browsing' }
'''

#Funções úteis
#Lê tracers e retorna lista de tuplas [ip, ip, tam]
def getDataFromFileColetivo(filename, header=False):
	fluxos = {}
	MH = {}
	with open(filename, 'r') as f:
		spamreader = f
		if filename.endswith('.csv'):
			spamreader = csv.reader(f, delimiter=',')
		for line in spamreader:
			l = line.replace('\n','').split('***')
			fluxos[l[0]] = np.asarray(l[1].replace('[','').replace(']','').split(','), dtype=float)
			MH[l[0]] = scipy.stats.hmean(fluxos[l[0]][fluxos[l[0]] > 0])
	return fluxos, MH

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
	return abs(np.linalg.norm(lista1 - lista2))	
	
def getDistanciaKullbackLeiber(lista1, lista2):
	result =  lista1 * np.log(lista1/lista2)
	#Isso aqui é pra evitar log(0). Substitui infinito por 0
	result[np.isposinf(result)] = 0
	return abs(np.nansum(result))
	#return scipy.stats.entropy(lista1, lista2)	
	
def getDistanciaWootters(lista1, lista2):
	return abs(np.arccos(np.sum(np.sqrt(lista1 * lista2))))
	
def getDistanciaHellinger(lista1, lista2):
	return abs(np.sqrt(0.5*np.sum((np.sqrt(lista1)-np.sqrt(lista2))**2)))
	
def getDistanciaJensen(lista1, lista2):
	#nansum ignora trata os casos de log errados ou divisões por zero como 0
	return abs(0.5*(np.nansum(lista1 * np.log(2.0*lista1/(lista1+lista2))) + np.nansum(lista2 * np.log(2.0*lista2/(lista1+lista2)))))
	
def getDistanciaBhattacharyya(lista1, lista2):
	return abs(max(0, np.log(np.sum(np.sqrt(lista1 * lista2)))))
	
def getDistanciaKolmogorovSmirnov(lista1, lista2):
	d,_ = scipy.stats.ks_2samp(lista1,lista2)
	return abs(d)

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
	return abs(D)
	
def getDistanciaQquadrado(lista1, lista2):
	return abs(np.nansum(np.ma.masked_invalid(((abs(lista1 - lista2) -0.5)*(abs(lista1 - lista2) -0.5))/lista2)))


def inList(args):
	global arquivoColetivo, protocoloAtual, PROTOCOLOS
	i = 0
	#Faz a validação com o ground truth (lista PROTOCOLOS), comparando se os ips de origem ou destino da tupla em questão realmente
	#pertencem ou não a classe da qual foi classificada
	for item in args:
		item = item.replace(']','').replace('[','').replace(' ','').replace("'",'').split(',')
		if PROTOCOLOS[arquivoColetivo].get(protocoloAtual) != None  and (item[0] 
in PROTOCOLOS[arquivoColetivo][protocoloAtual] or item[1] in PROTOCOLOS[arquivoColetivo][protocoloAtual]):
			i+=1
	return i
		
def classificaDistancias(nome, lista, mediasHarmonicas, tracersMisturados):
	global Qi, mediaConhecido, desvioConhecido
	
	#print 'Distância '+nome
	aceitos = []
	rejeitados = []

	# Loop para verificar os aceitos de acordo com as distâncias das amostras Qi na lista.
	# Se alguma condição de aceite não for satisfeita, considero como rejeitado
	for i in range(len(lista)):
		dist = lista[i]
		#Se o valor da distância for = 0, retorna a classe do tracer conhecido, toda freq. relativa em Qi entra na lista de aceitos
		if dist >= LIMITE_INFERIOR_ACEITA and dist <= LIMITE_SUPERIOR_ACEITA:
			aceitos.append(Qi[i])
		#Se o valor da distância estiver nesse intervalo, comparo se a média da freq. relativa de Qi está contida no 
		#intervalo [média - DP, média + DP] 
		elif dist > LIMITE_SUPERIOR_ACEITA and dist <= LIMITE_SUPERIOR_COMPARA: 
			if (np.mean(tracersMisturados[Qi[i]]) >= mediaConhecido - 2*desvioConhecido and np.mean(tracersMisturados[Qi[i]]) <= mediaConhecido + 2*desvioConhecido):
				aceitos.append(Qi[i])
			else:
				#Regra que verifica a média harmônica do fluxo na qual a freq. relativa pertence e a média da freq. relativa do tracer
				#conhecido em questão
				if mediasHarmonicas[Qi[i]] < mediaConhecido:
					aceitos.append(Qi[i])
				else:	
					rejeitados.append(Qi[i]) 
					
		#Se não entrou em algum caso passado, é automaticamente rejeitado
		else:
			#Regra que verifica a média harmônica do fluxo na qual a freq. relativa pertence e a média da freq. relativa do tracer
			#conhecido em questão
			if mediasHarmonicas[Qi[i]] < mediaConhecido:
				aceitos.append(Qi[i])
			else:	
				rejeitados.append(Qi[i])
	
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

	'''
	print 'Matriz de confusão -',nome
	print 'TP=', TP,' FP=', FP
	print 'FN=', FN,' TN=', TN
	print 'Accurácia: ', (TP + TN)/(1.0*len(aceitos) + len(rejeitados))
	precision = 0.0
	if (1.0*FP + TP) > 0 :
		precision = TP/(1.0*FP + TP) 
	print 'Precisão: ', precision
	recall = 0.0
	if (1.0*TP + FN) > 0:
		recall = TP/(1.0*TP + FN) 
	print 'Recall: ', recall
	denominador = precision + recall if precision + recall != 0 else 1
	fmeasure = (2*precision*recall)/denominador 
	print 'F-measure: ', fmeasure
	print '\n'	'''
	return [TP, TN, FP, FN]
		
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
	
	print 'Classe -', nomeProtocolo
	saida = 'Classe -'+nomeProtocolo+ '\n'
	
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
		f = open('Testes/Resultados_FA/Matriz Confusão '+nome +'-'+nomeProtocolo+'.txt', 'w')
		f.write(saida)
		f.close()




nomeProtocolo = ""
protocoloAtual = ""
arquivoColetivo = ""

if not os.path.exists('GraficosFR_EN/'):
	os.makedirs('GraficosFR_EN/')
	
#Prepara variáveis para matriz de confusão geral para cada distância por protocolo
matriz_eucl_global = []
matriz_leib_global = []
matriz_woot_global = []
matriz_hell_global = []
matriz_jens_global = []
matriz_bhat_global = []
matriz_kolm_global = []
matriz_maha_global = []
matriz_qqua_global = []

#Médias e DP dos tracers
mediaConhecido = 0 
desvioConhecido = 0
mediasHarmonicas = {}

individuaisFilenames = sorted(os.listdir('Individual_FR/'))

print 'Arquivos individuais encontrados:', individuaisFilenames, '\n'

# --------------------------------------------
for index in range(0, len(individuaisFilenames)):
	if ('.png' not in individuaisFilenames[index]):
		# Parte 1
		# Processamento dos tracers individuais
		tracerIndividual = individuaisFilenames[index]
		protocoloAtual = re.sub(r'(_labredes)?_\d+', '', tracerIndividual).replace('FR_','')
		protocoloAtual = re.sub(r'^[^_]+_', '', protocoloAtual).replace('_','')

		TRACER_INDIVIDUAL = 'Individual_FR/'+tracerIndividual
		freqRelativa, mediaConhecido, desvioConhecido = getDataFromFileIndividual(TRACER_INDIVIDUAL, header=True)
		
		plt.plot(range(100), freqRelativa)
		plt.ylim(top=1)
		plt.xlabel('Buckets')
		plt.ylabel('Relative Frequency')
		#plt.title(tracerIndividual)
		plt.savefig('GraficosFR_EN/'+tracerIndividual+'.png')
		plt.clf()


# ------------------------PARA FA, descomente abaixo ------------------


if not os.path.exists('GraficosFA_EN/'):
	os.makedirs('GraficosFA_EN/')
	
for index in range(0, len(individuaisFilenames)):
	if ('.png' not in individuaisFilenames[index]):
		# Parte 1
		# Processamento dos tracers individuais
		tracerIndividual = individuaisFilenames[index]
		protocoloAtual = re.sub(r'(_labredes)?_\d+', '', tracerIndividual).replace('FR_','')
		protocoloAtual = re.sub(r'^[^_]+_', '', protocoloAtual).replace('_','')

		TRACER_INDIVIDUAL = 'Individual_FR/'+tracerIndividual
		freqRelativa, mediaConhecido, desvioConhecido = getDataFromFileIndividual(TRACER_INDIVIDUAL, header=True)
		
		plt.plot(range(100), getFreqAcumulada(freqRelativa))
		plt.ylim(top=1)
		plt.xlabel('Buckets')
		plt.ylabel('Cumulative Frequency')
		#plt.title(tracerIndividual.replace('FR', 'FA'))
		plt.savefig('GraficosFA_EN/'+tracerIndividual.replace('FR', 'FA')+'.png')
		plt.clf()

