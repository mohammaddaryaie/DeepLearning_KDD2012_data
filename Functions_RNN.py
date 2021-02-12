from __future__ import print_function
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import tensorflow.keras.models as keras_models
from random import randrange
import matplotlib
from reportlab.graphics import samples
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math


Home_Address='.'
d_tokens = {}
tokens_idf = {}
sim_user_by_query=[]
sim_user_by_ad=[]
sim_user_by_keyword=[]
queries={}
keywords={}
document_number=150000000

def ifnull(var, val):
  if var is None or var == '':
    return val
  return var

def log_CTR(cl, imp):
  if imp  == 0.0:
    return 0.0
  return math.log10(cl+1/imp+1)

def Fill_idf_map():
	f_token = open(Home_Address+'/track2/DL_INPUT_EXTENDED_IDF.txt', 'r')
	for line in f_token:
		line = line.strip().split(',')
		try:
			tokens_idf[line[0]] = int(line[1])
		except:
			print('tokens_idf error:  ' + line)
	f_token.close()

def Fill_query_id_token_map():
	f_query = open(Home_Address+'/track2/main_data/query.txt', 'r')

	for line in f_query:
		line = line.strip().split('\t')
		try:
			queries[line[1]] = int(line[0])
		except:
			print('sim_user_by_query error:  ' + line)
	f_query.close()

def Fill_Keyword_id_token_map():
	f_query = open(Home_Address+'/track2/main_data/keyword.txt', 'r')

	for line in f_query:
		line = line.strip().split('\t')
		try:
			keywords[line[1]] = int(line[0])
		except:
			print('sim_user_by_keyword error:  ' + line)
	f_query.close()

def Fill_sim_user_by_query_ad_keyword():
	f_query = open(Home_Address+'/track2/sim_user_based_query_CI.txt', 'r')
	f_ad = open(Home_Address + '/track2/sim_user_based_ad_CI.txt', 'r')
	f_keyword = open(Home_Address + '/track2/sim_user_based_keyword_CI.txt', 'r')

	max_id_query = 26243605
	max_id_ad = 22647674
	max_id_keyword = 1249783

	for i in range(0, max_id_query + 1):
		sim_user_by_query.append([])
	for i in range(0, max_id_ad + 1):
		sim_user_by_ad.append([])
	for i in range(0, max_id_keyword + 1):
		sim_user_by_keyword.append([])

	for line in f_query:
		line = line.strip().split(',')
		try:
			sim_user_by_query[int(line[0])] = line[1] +':'+line[2]
		except:
			print('sim_user_by_query error:  ' + line)
	f_query.close()
	for line in f_ad:
		line = line.strip().split(',')
		try:
			sim_user_by_ad[int(line[0])] = line[1] +':'+line[2]
		except:
			print('sim_user_by_query error:  ' + line)
	f_ad.close()
	for line in f_keyword:
		line = line.strip().split(',')
		try:
			sim_user_by_keyword[int(line[0])] = line[1] +':'+line[2]
		except:
			print('sim_user_by_query_ad_keyword error:  ' + line)
	f_keyword.close()

def Compute_tfidf(term,line):
	line_list = list(line.strip().split('|'))
	tf = math.log10(line_list.count(term)+1)
	idf= math.log10(document_number/int('1' if term not in tokens_idf.keys() or tokens_idf[term]==0  else tokens_idf[term]))
	return float(tf*idf)

def Compute_cosine_sim(str1,str2):
	str1_list = str1.strip().split('|')
	str2_list = str2.strip().split('|')
	all_word = set(set(str1_list).union(set(str2_list)))
	l1=[]
	l2=[]
	for itm in all_word:
		l1.append(1) if itm in str1_list else l1.append(0)
		l2.append(1) if itm in str2_list else l2.append(0)
	numerator = sum(l1[x] * l2[x] for x in range(len(all_word)))
	sum1 = sum(l1[x] ** 2 for x in range(len(l1)) )
	sum2 = sum(l2[x] ** 2 for x in range(len(l2)))
	denominator = math.sqrt(sum1*sum2)
	if denominator==0:
		return 0.0
	else:
		return round(numerator/denominator,3)

def Compute_cosine_sim_TFIDF(str1,str2):
	str1_list = str1.strip().split('|')
	str2_list = str2.strip().split('|')
	all_word = set(set(str1_list).union(set(str2_list)))
	l1=[];l2=[]
	for itm in all_word:
		l1.append(Compute_tfidf(itm,str1)) if itm in str1_list else l1.append(0)
		l2.append(Compute_tfidf(itm,str2)) if itm in str2_list else l2.append(0)
	numerator = sum(l1[x] * l2[x] for x in range(len(all_word)))
	sum1 = sum(l1[x] ** 2 for x in range(len(l1)) )
	sum2 = sum(l2[x] ** 2 for x in range(len(l2)))
	denominator = math.sqrt(sum1*sum2)
	if denominator==0:
		return 0.0
	else:
		return round(numerator/denominator,3)

def Fill_List_Input(line):
	line_error = line
	x=[]
	line = line.strip().split(",")
	label = line[0]
	col_num = 29

	#input_dl_matrix = np.zeros((1, col_num)).tolist()
	External_list = range(0,col_num)
	for n in External_list:
		x.append(0)
	try:
		# click=0,impression=1,displayurl=2,adid=3,advertiserid=4,depth=5,position=6,queryid=7,
		# keywordid=8,titleid=9,description=10,user_gender=11,user_group=12,userid=13,
		# pos_user_rate(click:impression)=14,dep_user_rate=15,
		# user_rate=16,ad_rate=17,adv_rate=18,url_rate=19,pos_ad_rate=20,dep_ad_rate=21
		# gen_ad_rate=22,gen_adv_rate=23,group_ad_rate=24
		# tfidf is better than sim: tfidf(q7-k8)=25,tfidf(q7-t9)=26,tfidf(q7-d10)=27,tfidf(k8-t9)=28,tfidf(k8-d10)=29,tfidf(t9-d10)=30
		# should be removed : sim(q7-k8)=31,sim(q7-t9)=32,sim(q7-d10)=33,sim(k8-t9)=34,sim(k8-d10)=35,sim(t9-d10)=36
		x[0] = float(line[1])
		x[1] = float(line[5])
		x[2] = float(line[6])
		x[3] = float(line[11])
		x[4] = float(line[12])
		x[5] = 0 if float(line[14].split(':')[1]) < 0 else float(line[14].split(':')[0])
		x[6] = 0 if float(line[14].split(':')[1]) < 0 else float(line[14].split(':')[1])
		x[7] = 0 if float(line[15].split(':')[1]) < 0 else float(line[15].split(':')[0])
		x[8] = 0 if float(line[15].split(':')[1]) < 0 else float(line[15].split(':')[1])
		#user rate
		if float(line[16].split(':')[1]) > 5 :
			x[9] = float(line[16].split(':')[0])
		else:
			x[9] = 0 if len(sim_user_by_query[queries[line[7]]])==0 else float(sim_user_by_query[queries[line[7]]].split(':')[0])

		if float(line[16].split(':')[1]) > 5:
			x[10] =float(line[16].split(':')[1])
		else:
			x[10] = 0 if len(sim_user_by_query[queries[line[7]]])==0 else float(sim_user_by_query[queries[line[7]]].split(':')[1])

		x[11] = 0 if float(line[17].split(':')[1]) < 0 else float(line[17].split(':')[0])
		x[12] = 0 if float(line[17].split(':')[1]) < 0 else float(line[17].split(':')[1])
		x[13] = 0 if float(line[18].split(':')[1]) < 0 else float(line[18].split(':')[0])
		x[14] = 0 if float(line[18].split(':')[1]) < 0 else float(line[18].split(':')[1])
		x[15] = 0 if float(line[19].split(':')[1]) < 0 else float(line[19].split(':')[0])
		x[16] = 0 if float(line[19].split(':')[1]) < 0 else float(line[19].split(':')[1])
		x[17] = float(line[25])
		x[18] = float(line[26])
		x[19] = float(line[27])
		x[20] = float(line[28])
		x[21] = float(line[29])
		x[22] = float(line[30])
		x[23] = 0 if len(sim_user_by_query[queries[line[7]]])==0 else float(sim_user_by_query[queries[line[7]]].split(':')[0])
		x[24] = 0 if len(sim_user_by_query[queries[line[7]]])==0 else float(sim_user_by_query[queries[line[7]]].split(':')[1])
		x[25] = 0 if len(sim_user_by_ad[int(line[3])])==0 else float(sim_user_by_ad[int(line[3])].split(':')[0])
		x[26] = 0 if len(sim_user_by_ad[int(line[3])])==0 else float(sim_user_by_ad[int(line[3])].split(':')[1])
		x[27] = 0 if len(sim_user_by_keyword[keywords[line[8]]]) == 0 else float(sim_user_by_keyword[keywords[line[8]]].split(':')[0])
		x[28] = 0 if len(sim_user_by_keyword[keywords[line[8]]]) == 0 else float(sim_user_by_keyword[keywords[line[8]]].split(':')[1])
	except Exception as e:
		print('Error: ' + line_error+'\n' + str(e))


	#x.append(input_dl_matrix.tolist())
	return np.array(x, dtype=np.float32)
