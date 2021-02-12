from builtins import str
import tensorflow
from tensorflow.keras.models import Sequential
from numpy import loadtxt
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy

Home_Address='.'
def ifnull(var, val):
  if var is None or var=='':
    return val
  return var

# Number:1, Description: replace ids with values in training datasset, output: DL_INPUT.txt
f_k = open(Home_Address+'/track2/main_data/keyword.txt', 'r')
f_d = open(Home_Address+'/track2/main_data/description.txt', "r")
f_q = open(Home_Address+'/track2/main_data/query.txt', "r")
f_t = open(Home_Address+'/track2/main_data/title.txt', "r")
f_u = open(Home_Address+'/track2/main_data/user.txt', "r")
def ifnull(var, val):
  if var is None or var == '':
    return val
  return var
d_k={}
d_d={}
d_q={}
d_t={}
d_u={}

for line in f_k:
    line = line.strip().split('\t')
    d_k[int(line[0])]=line[1]

for line in f_d:
    line = line.strip().split('\t')
    d_d[int(line[0])]=line[1]

for line in f_q:
    line = line.strip().split('\t')
    d_q[int(line[0])]=line[1]

for line in f_t:
    line = line.strip().split('\t')
    d_t[int(line[0])]=line[1]

for line in f_u:
    line = line.strip().split('\t')
    d_u[int(line[0])]=line[1]+','+line[2]

f_tr = open(Home_Address+"/track2/main_data/training.txt", "r")
fs = open(Home_Address+"/track2/DL_INPUT.txt", "w")

counter=0
for line in f_tr:
    counter=counter+1
    if 0 == counter%10000000:
        print(counter)
    line = line.strip().split('\t')
    try:
        fs.write(line[0]+','+line[1]+','+line[2]+','+line[3]+','+line[4]+','+line[5]+','+line[6]+','+ifnull(d_q[int(line[7])],'-100')+','+ifnull(d_k[int(line[8])],'-100')+','+ifnull(d_t[int(line[9])],'-100')+','+ifnull(d_d[int(line[10])],'-100')+','+ifnull(d_u[int(line[11])],'0,0')+','+ifnull(line[11],'-100')+'\r\n')
    except:
        fs.write(line[0] + ',' + line[1] + ',' + line[2] + ',' + line[3] + ',' + line[4] + ',' + line[5] + ',' + line[6] + ',' + d_q[int(line[7])] + ',' + d_k[int(line[8])] + ',' + d_t[int(line[9])] + ',' + d_d[int(line[10])] + ',' + '0,0'+','+ifnull(line[11],'-100') + '\r\n')
        #print(str(counter)+','+line[0]+','+line[1]+','+line[2]+','+line[3]+','+line[4]+','+line[5]+','+line[6]+','+line[7]+','+line[8]+','+line[9]+','+line[10]+','+line[11]+'\r\n')

f_d.close()
f_k.close()
f_q.close()
f_t.close()
f_u.close()
f_tr.close()
fs.close()

del d_k
del d_d
del d_q
del d_t
del d_u

# Number:2, Description:  Build dictionary of tokenids by Q, k, T and D files for matrix index output: tokens.txt
f_k = open(Home_Address+'/track2/keyword.txt', 'r')
f_d = open(Home_Address+"/track2/description.txt", "r")
f_q = open(Home_Address+"/track2/query.txt", "r")
f_t = open(Home_Address+"/track2/title.txt", "r")

fs = open(Home_Address+"/track2/tokens.txt", "w")
token_list = []
for line in f_k:
    line = line.strip().split('\t')
    token = line[1].strip().split('|')
    for each_token in token:
        token_list.append(int(each_token))
token_list=list(set(token_list))
for line in f_d:
    line = line.strip().split('\t')
    token = line[1].strip().split('|')
    for each_token in token:
        token_list.append(int(each_token))
token_list=list(set(token_list))
for line in f_q:
    line = line.strip().split('\t')
    token = line[1].strip().split('|')
    for each_token in token:
        token_list.append(int(each_token))

token_list=list(set(token_list))
for line in f_t:
    line = line.strip().split('\t')
    token = line[1].strip().split('|')
    for each_token in token:
        token_list.append(int(each_token))

token_list=list(set(token_list))
token_list.sort()
counter=0
for line in token_list:
    fs.write(str(counter)+','+str(line) + "\n")
    counter = counter+1

f_d.close()
f_q.close()
f_t.close()
f_k.close()
fs.close()

# Number:3, Description: Add dictionary of ids by training data for matrix index output:tokens.txt
f_t = open(Home_Address+"/track2/training.txt", "r")
token_list = []
counter=0
for line in f_t:
    counter = counter + 1
    if counter %1000000 ==0:
        print(str(counter))
        token_list = list(set(token_list))
    line = line.strip().split('\t')
    token = 'u'+line[2] #displayurl
    token_list.append(token)
    token = 'ad' + line[3]  #adid
    token_list.append(token)
    token = 'av' + line[4]  #advertiserid
    token_list.append(token)

f_t.close()

token_list=list(set(token_list))
token_list.sort()

ListOfTokens = pd.read_csv(Home_Address+'/track2/tokens.txt')
counter=ListOfTokens["0"].max()+1
del ListOfTokens

fs = open(Home_Address+"/track2/tokens.txt", "a+")

for line in fs:
    if line=="":
        break

for line in token_list:
    fs.write(str(counter)+','+line + "\n")
    counter = counter+1

del token_list
fs.close()

# Number:4, Description: Extract click and impression of each id,
# output: user_CI, q_CI, ad_CI, adv_CI, url_CI, gen_adv_CI, gen_ad_CI, group_ad_CI, dep_user_CI, dep_ad_CI
Home_Address='/media/fs_Linux_Files/'
f_user = open(Home_Address+'/track2/user_CI.txt', 'w')
f_ad = open(Home_Address+'/track2/ad_CI.txt', "w")
f_q = open(Home_Address+'/track2/q_CI.txt', "w")
f_adv = open(Home_Address+'/track2/adv_CI.txt', "w")
f_url = open(Home_Address+'/track2/url_CI.txt', "w")
f_gen_adv = open(Home_Address+'/track2/gen_adv_CI.txt', "w")
f_gen_ad = open(Home_Address+'/track2/gen_ad_CI.txt', "w")
f_group_ad = open(Home_Address+'/track2/group_ad_CI.txt', "w")

f_r_user = open(Home_Address+'/track2/main_data/user.txt', "r")
f_tr = open(Home_Address+'/track2/main_data/training.txt', "r")

def ifnull(var, val):
  if var is None or var == '':
    return val
  return var
d_user={}
d_ad ={}
d_q ={}
d_adv={}
d_url={}
d_gen_adv = {}
d_gen_ad = {}
d_group_ad={}

d_his_user_gen={}
d_his_user_group={}
for line in f_r_user:
    line = line.strip().split('\t')
    d_his_user_gen[line[0]]=line[1]
    d_his_user_group[line[0]] = line[2]

#click=0,impression=1,displayurl=2,adid=3,advertiserid=4,depth=5,position=6,queryid=7,
# keywordid=8,titleid=9,description=10,userid=11
counter=0
for line in f_tr:
    counter=counter+1
    if 0 == counter%10000000:
        print(counter)
    line = line.strip().split('\t')
    if line[2] not in d_url:
        d_url[line[2]]=line[0]+','+line[1]
    else:
        click=int(d_url[line[2]].split(',')[0])+int(line[0])
        impression = int(d_url[line[2]].split(',')[1]) + int(line[1])
        d_url[line[2]] =str(click) + ',' + str(impression)

    if line[3] not in d_ad:
        d_ad[line[3]] = line[0] + ',' + line[1]
    else:
        click = int(d_ad[line[3]].split(',')[0]) + int(line[0])
        impression = int(d_ad[line[3]].split(',')[1]) + int(line[1])
        d_ad[line[3]] = str(click) + ',' + str(impression)

    if line[7] not in d_q:
        d_q[line[7]] = line[0] + ',' + line[1]
    else:
        click = int(d_q[line[7]].split(',')[0]) + int(line[0])
        impression = int(d_q[line[7]].split(',')[1]) + int(line[1])
        d_q[line[7]] = str(click) + ',' + str(impression)

    if line[4] not in d_adv:
        d_adv[line[4]] = line[0] + ',' + line[1]
    else:
        click = int(d_adv[line[4]].split(',')[0]) + int(line[0])
        impression = int(d_adv[line[4]].split(',')[1]) + int(line[1])
        d_adv[line[4]] = str(click) + ',' + str(impression)

    if line[11] not in d_user:
        d_user[line[11]] = line[0] + ',' + line[1]
    else:
        click = int(d_user[line[11]].split(',')[0]) + int(line[0])
        impression = int(d_user[line[11]].split(',')[1]) + int(line[1])
        d_user[line[11]] = str(click) + ',' + str(impression)


    gen_ad_index= d_his_user_gen[line[11]] if line[11] in d_his_user_gen else '-100'
    if gen_ad_index != '-100' and d_his_user_gen[line[11]]+':'+line[3] not in d_gen_ad:
        d_gen_ad[d_his_user_gen[line[11]]+':'+line[3]] = line[0] + ',' + line[1]
    elif gen_ad_index != '-100':
        click = int(d_gen_ad[d_his_user_gen[line[11]]+':'+line[3]].split(',')[0]) + int(line[0])
        impression = int(d_gen_ad[d_his_user_gen[line[11]]+':'+line[3]].split(',')[1]) + int(line[1])
        d_gen_ad[d_his_user_gen[line[11]]+':'+line[3]] = str(click) + ',' + str(impression)

    gen_adv_index= d_his_user_gen[line[11]] if line[11] in d_his_user_gen else '-100'
    if gen_adv_index != '-100' and d_his_user_gen[line[11]]+':'+line[4] not in d_gen_adv:
        d_gen_adv[d_his_user_gen[line[11]]+':'+line[4]] = line[0] + ',' + line[1]
    elif gen_adv_index != '-100':
        click = int(d_gen_adv[d_his_user_gen[line[11]]+':'+line[4]].split(',')[0]) + int(line[0])
        impression = int(d_gen_adv[d_his_user_gen[line[11]]+':'+line[4]].split(',')[1]) + int(line[1])
        d_gen_adv[d_his_user_gen[line[11]]+':'+line[4]] = str(click) + ',' + str(impression)

    group_ad_index= d_his_user_group[line[11]] if line[11] in d_his_user_group else '-100'
    if group_ad_index != '-100' and d_his_user_group[line[11]]+':'+line[3] not in d_group_ad:
        d_group_ad[d_his_user_group[line[11]]+':'+line[3]] = line[0] + ',' + line[1]
    elif group_ad_index != '-100':
        click = int(d_group_ad[d_his_user_group[line[11]]+':'+line[3]].split(',')[0]) + int(line[0])
        impression = int(d_group_ad[d_his_user_group[line[11]]+':'+line[3]].split(',')[1]) + int(line[1])
        d_group_ad[d_his_user_group[line[11]]+':'+line[3]] = str(click) + ',' + str(impression)


for item in d_ad.keys():
    f_ad.write(item+','+d_ad[item]+'\r\n')

for item in d_q.keys():
    f_q.write(item+','+d_q[item]+'\r\n')

for item in d_adv.keys():
    f_adv.write(item+','+d_adv[item]+'\r\n')

for item in d_user.keys():
    f_user.write(item+','+d_user[item]+'\r\n')

for item in d_url.keys():
    f_url.write(item+','+d_url[item]+'\r\n')

for item in d_gen_adv.keys():
    f_gen_adv.write(item+','+d_gen_adv[item]+'\r\n')

for item in d_gen_ad.keys():
    f_gen_ad.write(item+','+d_gen_ad[item]+'\r\n')

for item in d_group_ad.keys():
    f_group_ad.write(item+','+d_group_ad[item]+'\r\n')

f_user.close()
f_tr.close()
f_ad.close()
f_q.close()
f_adv.close()
f_url.close()
f_gen_adv.close()
f_gen_ad.close()
f_group_ad.close()


del d_user
del d_ad
del d_q
del d_adv
del d_url
del d_gen_ad
del d_gen_adv
del d_group_ad

f_dep_user = open(Home_Address+'/track2/dep_user_CI.txt', 'w')
f_dep_ad = open(Home_Address+"/track2/dep_ad_CI.txt", "w")

f_tr = open(Home_Address+"/track2/main_data/training.txt", "r")

def ifnull(var, val):
  if var is None or var == '':
    return val
  return var

d_dep_user = {}
d_dep_ad = {}

#click=0,impression=1,displayurl=2,adid=3,advertiserid=4,depth=5,position=6,queryid=7,keywordid=8,titleid=9,description=10,userid=11
counter=0
for line in f_tr:
    counter=counter+1
    if 0 == counter%10000000:
        print(counter)
    line = line.strip().split('\t')

    if line[5]+':'+line[11] not in d_dep_user:
        d_dep_user[line[5]+':'+line[11]] = line[0] + ',' + line[1]
    else:
        click = int(d_dep_user[line[5]+':'+line[11]].split(',')[0]) + int(line[0])
        impression = int(d_dep_user[line[5]+':'+line[11]].split(',')[1]) + int(line[1])
        d_dep_user[line[5]+':'+line[11]] = str(click) + ',' + str(impression)

    if line[5]+':'+line[3] not in d_dep_ad:
        d_dep_ad[line[5]+':'+line[3]] = line[0] + ',' + line[1]
    else:
        click = int(d_dep_ad[line[5]+':'+line[3]].split(',')[0]) + int(line[0])
        impression = int(d_dep_ad[line[5]+':'+line[3]].split(',')[1]) + int(line[1])
        d_dep_ad[line[5]+':'+line[3]] = str(click) + ',' + str(impression)

for item in d_dep_ad.keys():
    f_dep_ad.write(item+','+d_dep_ad[item]+'\r\n')

for item in d_dep_user.keys():
    f_dep_user.write(item+','+d_dep_user[item]+'\r\n')

f_dep_user.close()
f_dep_ad.close()

del d_dep_ad
del d_dep_user

f_pos_user = open(Home_Address+'/track2/pos_user_CI.txt', 'w')
f_pos_ad = open(Home_Address+"/track2/pos_ad_CI.txt", "w")

f_tr = open(Home_Address+"/track2/main_data/training.txt", "r")

def ifnull(var, val):
  if var is None or var == '':
    return val
  return var

d_pos_user = {}
d_pos_ad = {}

#click=0,impression=1,displayurl=2,adid=3,advertiserid=4,depth=5,position=6,queryid=7,
# keywordid=8,titleid=9,description=10,userid=11
counter=0
for line in f_tr:
    counter=counter+1
    if 0 == counter%10000000:
        print(counter)
    line = line.strip().split('\t')

    if line[6]+':'+line[11] not in d_pos_user:
        d_pos_user[line[6]+':'+line[11]] = line[0] + ',' + line[1]
    else:
        click = int(d_pos_user[line[6]+':'+line[11]].split(',')[0]) + int(line[0])
        impression = int(d_pos_user[line[6]+':'+line[11]].split(',')[1]) + int(line[1])
        d_pos_user[line[6]+':'+line[11]] = str(click) + ',' + str(impression)

    if line[6]+':'+line[3] not in d_pos_ad:
        d_pos_ad[line[6]+':'+line[3]] = line[0] + ',' + line[1]
    else:
        click = int(d_pos_ad[line[6]+':'+line[3]].split(',')[0]) + int(line[0])
        impression = int(d_pos_ad[line[6]+':'+line[3]].split(',')[1]) + int(line[1])
        d_pos_ad[line[6]+':'+line[3]] = str(click) + ',' + str(impression)

for item in d_pos_ad.keys():
    f_pos_ad.write(item+','+d_pos_ad[item]+'\r\n')

for item in d_pos_user.keys():
    f_pos_user.write(item+','+d_pos_user[item]+'\r\n')

f_pos_ad.close()
f_pos_user.close()

del d_pos_ad
del d_pos_user

# Number:5, Description: Find similar users based on query and ad,
# output: sim_user_based_keyword.txt, sim_user_based_query.txt, sim_user_based_ad.txt,
# sim_user_based_query_CI.txt, sim_user_based_ad_CI.txt, sim_user_based_keyword_CI.txt
Home_Address='/media/fs_Linux_Files/'
f_r = open(Home_Address+'track2/main_data/training.txt', "r")
f_w_q = open(Home_Address+'track2/sim_user_based_query.txt', "w")
f_w_ad = open(Home_Address+'track2/sim_user_based_ad.txt', "w")
f_w_k = open(Home_Address+'track2/sim_user_based_keyword.txt', "w")
user_ad_d=[]
user_q_d=[]
user_k_d=[]

max_id_query=26243605
max_id_ad=22647674
max_id_user=23907634
max_id_keyword=1249783

#click=0,impression=1,displayurl=2,adid=3,advertiserid=4,depth=5,position=6,queryid=7,
# keywordid=8,titleid=9,description=10,userid=11

'''
counter=0
for line in f_r:
    line=line.strip().split('\t')
    counter=counter+1
    if 0 == counter%1000000:
        print(counter)
    if int(line[7])>max_id_query:
        max_id_query=int(line[7])
    if int(line[3])>max_id_ad:
        max_id_ad=int(line[7])
    if int(line[11])>max_id_user:
        max_id_user=int(line[11])
    if int(line[8])>max_id_keyword:
        max_id_keyword=int(line[8])
'''
for i in range(0,max_id_query+1):
    user_q_d.append([])

for i in range(0,max_id_ad+1):
    user_ad_d.append([])

for i in range(0,max_id_keyword+1):
    user_k_d.append([])
#Compute Query
counter=0
f_r = open(Home_Address+'track2/main_data/training.txt', "r")
for line in f_r:
    line=line.strip().split('\t')
    counter=counter+1
    if 0 == counter%1000000:
        print(counter)
    if 0 == counter % 20000000:
        for i in range(0,len(user_q_d)):
            user_q_d[i]=list(set(user_q_d[i]))
    user_q_d[int(line[7])].append(int(line[11]))
f_r.close()

for item in range(0,len(user_q_d)):
    f_w_q.write(str(item)+','+str(user_q_d[item]).replace('[','').replace(']','')+'\n')
f_w_q.close()
del user_q_d
#Compute ad
counter=0
f_r = open(Home_Address+'track2/main_data/training.txt', "r")
for line in f_r:
    line=line.strip().split('\t')
    counter=counter+1
    if 0 == counter%1000000:
        print(counter)
    if 0 == counter % 20000000:
        for i in range(0,len(user_ad_d)):
            user_ad_d[i] = list(set(user_ad_d[i]))
    user_ad_d[int(line[3])].append(int(line[11]))
f_r.close()

for item in range(0,len(user_ad_d)):
    f_w_ad.write(str(item)+','+str(user_ad_d[item]).replace('[','').replace(']','')+'\n')
f_w_ad.close()
del user_ad_d
#Compute Keyword
counter=0
f_r = open(Home_Address+'track2/main_data/training.txt', "r")
for line in f_r:
    line=line.strip().split('\t')
    counter=counter+1
    if 0 == counter%1000000:
        print(counter)
    if 0 == counter % 20000000:
        for i in range(0,len(user_k_d)):
            user_k_d[i] = list(set(user_k_d[i]))
    user_k_d[int(line[8])].append(int(line[11]))
f_r.close()

for item in range(0,len(user_k_d)):
    f_w_k.write(str(item)+','+str(user_k_d[item]).replace('[','').replace(']','')+'\n')
f_w_k.close()
del user_k_d

#Find click and impression number for each query and ad by their users
Home_Address='/media/fs_Linux_Files/'
f_user_r = open(Home_Address+'track2/user_CI.txt', "r")
f_q_r = open(Home_Address+'track2/sim_user_based_query.txt', "r")
f_ad_r = open(Home_Address+'track2/sim_user_based_ad.txt', "r")
f_k_r = open(Home_Address+'track2/sim_user_based_keyword.txt', "r")

f_q_w = open(Home_Address+'track2/sim_user_based_query_CI.txt', "w")
f_ad_w = open(Home_Address+'track2/sim_user_based_ad_CI.txt', "w")
f_k_w = open(Home_Address+'track2/sim_user_based_keyword_CI.txt', "w")
user_ad_d=[]
user_q_d=[]
user_u_d=[]
user_k_d=[]

max_id_query=26243605
max_id_ad=22647674
max_id_user=23907634
max_id_keyword=1249783

for i in range(0,max_id_user+1):
    user_u_d.append([])

for line in f_user_r:
    line=line.strip().split(',')
    user_u_d[int(line[0])]=line[1]+','+line[2]
#Extract Query Click and Impression
for i in range(0,max_id_query+1):
    user_q_d.append([])

for line in f_q_r:
    line=line.strip().split(',')
    user_list= line[1:len(line)]
    count_users=0
    sum_click=0
    sum_imp=0
    for user in user_list:
        if user not in ('0',''):
            click=user_u_d[int(user)].strip().split(',')[0]
            impression=user_u_d[int(user)].strip().split(',')[1]
            if int(impression)> 5:
                count_users+=1
                sum_click+=int(click)
                sum_imp+=int(impression)
    if count_users==0:
        count_users=1
    if sum_imp>0:
        user_q_d[int(line[0])]=str(round(sum_click/count_users,3))+','+str(round(sum_imp/count_users,3))

for i in range(0,len(user_q_d)):
    if len(user_q_d[i])>0:
        f_q_w.write(str(i)+','+user_q_d[i]+'\n')
f_q_w.close()
del user_q_d
#Extract Ad Click and Impression
for i in range(0,max_id_ad+1):
    user_ad_d.append([])
for line in f_ad_r:
    line=line.strip().split(',')
    user_list= line[1:len(line)]
    count_users=0
    sum_click=0
    sum_imp=0
    for user in user_list:
        if user not in ('0',''):
            click=user_u_d[int(user)].strip().split(',')[0]
            impression=user_u_d[int(user)].strip().split(',')[1]
            if int(impression)> 5:
                count_users+=1
                sum_click+=int(click)
                sum_imp+=int(impression)
    if count_users==0:
        count_users=1
    if sum_imp>0:
        user_ad_d[int(line[0])]=str(round(sum_click/count_users,3))+','+str(round(sum_imp/count_users,3))

for i in range(0,len(user_ad_d)):
    if len(user_ad_d[i])>0:
        f_ad_w.write(str(i)+','+user_ad_d[i]+'\n')
f_ad_w.close()
del user_ad_d
#Extract Keyword Click and Impression
for i in range(0,max_id_keyword+1):
    user_k_d.append([])
for line in f_k_r:
    line=line.strip().split(',')
    user_list= line[1:len(line)]
    count_users=0
    sum_click=0
    sum_imp=0
    for user in user_list:
        if user not in ('0',''):
            click=user_u_d[int(user)].strip().split(',')[0]
            impression=user_u_d[int(user)].strip().split(',')[1]
            if int(impression)> 5:
                count_users+=1
                sum_click+=int(click)
                sum_imp+=int(impression)
    if count_users==0:
        count_users=1
    if sum_imp>0:
        user_k_d[int(line[0])]=str(round(sum_click/count_users,3))+','+str(round(sum_imp/count_users,3))

for i in range(0,len(user_k_d)):
    if len(user_k_d[i])>0:
        f_k_w.write(str(i)+','+user_k_d[i]+'\n')
f_k_w.close()
del user_k_d

# Number:6, Description: Add probabilities and similarity to train data,
# output: DL_INPUT_measure_user.txt, DL_INPUT_measure.txt
Home_Address='/media/fs_Linux_Files/'
import Functions_RNN
Functions_RNN.Fill_idf_map()
f_pos_user = open(Home_Address+"/track2/pos_user_CI.txt", "r")
f_dep_user = open(Home_Address+"/track2/dep_user_CI.txt", "r")

f_tr = open(Home_Address+"/track2/DL_INPUT.txt", "r")
fs = open(Home_Address+"/track2/DL_INPUT_measure_user.txt", "w")
def ifnull(var, val):
  if var is None or var == '':
    return val
  return var

d_pos_user = {}
d_dep_user = {}

for line in f_pos_user:
    line = line.strip().split(',')
    d_pos_user[int(line[0].split(':')[0]+'00'+line[0].split(':')[1])]=line[1]+':'+line[2]

for line in f_dep_user:
    line = line.strip().split(',')
    d_dep_user[int(line[0].split(':')[0]+'00'+line[0].split(':')[1])]=line[1]+':'+line[2]

#click=0,impression=1,displayurl=2,adid=3,advertiserid=4,depth=5,position=6,queryid=7,
# keywordid=8,titleid=9,description=10,user_gender=11,user_group=12,userid=13,
# pos_user_rate(click:impression)=14,dep_user_rate=15

counter=0
for line in f_tr:
    counter=counter+1
    if 0 == counter%10000000:
        print(counter)
    main_line=line.strip()
    line = line.strip().split(',')
    fs.write(main_line+','+d_pos_user[int(line[6]+'00'+line[13])]+','+\
             d_dep_user[int(line[5]+'00'+line[13])]+'\r\n')

f_pos_user.close()
f_dep_user.close()
f_tr.close()
fs.close()

del d_pos_user
del d_dep_user

Home_Address='/media/fs_Linux_Files/'
import Functions_RNN
Functions_RNN.Fill_idf_map()
f_user = open(Home_Address+'/track2/user_CI.txt', 'r')
f_ad = open(Home_Address+"/track2/ad_CI.txt", "r")
f_adv = open(Home_Address+"/track2/adv_CI.txt", "r")
f_url = open(Home_Address+"/track2/url_CI.txt", "r")
f_pos_ad = open(Home_Address+'/track2/pos_ad_CI.txt', 'r')
f_dep_ad = open(Home_Address+"/track2/dep_ad_CI.txt", "r")
f_gen_ad = open(Home_Address+'/track2/gen_ad_CI.txt', 'r')
f_gen_adv = open(Home_Address+"/track2/gen_adv_CI.txt", "r")
f_group_ad = open(Home_Address+"/track2/group_ad_CI.txt", "r")

f_tr = open(Home_Address+'/track2/DL_INPUT_measure_user.txt', "r")
fs = open(Home_Address+'/track2/DL_INPUT_measure.txt', "w")

def ifnull(var, val):
  if var is None or var == '':
    return val
  return var
d_user={}
d_ad={}
d_adv={}
d_url={}

d_pos_ad = {}
d_dep_ad = {}
d_gen_ad = {}
d_gen_adv = {}
d_group_ad = {}


for line in f_user:
    line = line.strip().split(',')
    d_user[line[0]]=line[1]+':'+line[2]

for line in f_ad:
    line = line.strip().split(',')
    d_ad[line[0]]=line[1]+':'+line[2]

for line in f_adv:
    line = line.strip().split(',')
    d_adv[line[0]]=line[1]+':'+line[2]

for line in f_url:
    line = line.strip().split(',')
    d_url[line[0]]=line[1]+':'+line[2]

for line in f_pos_ad:
    line = line.strip().split(',')
    d_pos_ad[line[0]]=line[1]+':'+line[2]

for line in f_dep_ad:
    line = line.strip().split(',')
    d_dep_ad[line[0]]=line[1]+':'+line[2]

for line in f_gen_ad:
    line = line.strip().split(',')
    d_gen_ad[line[0]]=line[1]+':'+line[2]

for line in f_gen_adv:
    line = line.strip().split(',')
    d_gen_adv[line[0]]=line[1]+':'+line[2]

for line in f_group_ad:
    line = line.strip().split(',')
    d_group_ad[line[0]]=line[1]+':'+line[2]

#click=0,impression=1,displayurl=2,adid=3,advertiserid=4,depth=5,position=6,queryid=7,
# keywordid=8,titleid=9,description=10,user_gender=11,user_group=12,userid=13,
# pos_user_rate(click:impression)=14,dep_user_rate=15,
# user_rate=16,ad_rate=17,adv_rate=18,url_rate=19,pos_ad_rate=20,dep_ad_rate=21
# gen_ad_rate=22,gen_adv_rate=23,group_ad_rate=24
#tfidf(q7-k8)=25,tfidf(q7-t9)=26,tfidf(q7-d10)=27,tfidf(k8-t9)=28,tfidf(k8-d10)=29,tfidf(t9-d10)=30
#sim(q7-k8)=31,sim(q7-t9)=32,sim(q7-d10)=33,sim(k8-t9)=34,sim(k8-d10)=35,sim(t9-d10)=36

counter=0
for line in f_tr:
    counter=counter+1
    if 0 == counter%1000000:
        print(counter)
    main_line=line.strip()
    line = line.strip().split(',')

    complete_line=ifnull(d_user[line[13]],'0:0')+','+ifnull(d_ad[line[3]],'0:0')+','\
                  +ifnull(d_adv[line[4]],'0:0')+','+ifnull(d_url[line[2]],'0:0')+',' \
                  + ifnull(d_pos_ad[line[6]+':'+line[3]], '0:0') + ','  \
                  + ifnull(d_dep_ad[line[5]+':'+line[3]], '0:0') + ',' \
                  + ifnull('0:0' if line[11]+':'+line[3] not in d_gen_ad else d_gen_ad[line[11]+':'+line[3]], '0:0') + ',' \
                  + ifnull('0:0' if line[11]+':'+line[4] not in d_gen_adv else d_gen_adv[line[11]+':'+line[4]], '0:0') + ',' \
                  + ifnull('0:0' if line[12]+':'+line[3] not in d_group_ad else d_group_ad[line[12]+':'+line[3]], '0:0') + ','  \
                  +str(Functions_RNN.Compute_cosine_sim_TFIDF(line[7],line[8]))+','\
                  + str(Functions_RNN.Compute_cosine_sim_TFIDF(line[7], line[9])) + ','\
                  + str(Functions_RNN.Compute_cosine_sim_TFIDF(line[7], line[10])) + ','\
                  + str(Functions_RNN.Compute_cosine_sim_TFIDF(line[8], line[9])) + ','\
                  + str(Functions_RNN.Compute_cosine_sim_TFIDF(line[8], line[10])) + ','\
                  + str(Functions_RNN.Compute_cosine_sim_TFIDF(line[9], line[10])) + ','\
                  + str(Functions_RNN.Compute_cosine_sim(line[7], line[8])) + ','\
                  + str(Functions_RNN.Compute_cosine_sim(line[7], line[9])) + ','\
                  + str(Functions_RNN.Compute_cosine_sim(line[7], line[10])) + ','\
                  + str(Functions_RNN.Compute_cosine_sim(line[8], line[9])) + ','\
                  + str(Functions_RNN.Compute_cosine_sim(line[8], line[10])) + ','\
                  + str(Functions_RNN.Compute_cosine_sim(line[9], line[10]))\
                  +'\r\n'
    if int(line[0]) > 1:
        for i in range(int(line[0])):
            if i == int(line[0]) - 1:
                fs.write('1' + ',' + str(int(line[1]) - int(line[0]) +1) + ','+ line[2]+ ','+line[3]+ ','
                         +line[4]+ ','+line[5]+ ',' +line[6]+ ',' +line[7]+ ',' +line[8]+ ',' +
                         line[9]+ ',' +line[10]+ ','+line[11]+ ','+line[12]+ ','+line[13]+ ','+
                         line[14]+ ',' +line[15]+ ','+
                         complete_line)
            else:
                fs.write('1' + ',' + '1' + ',' +  line[2]+ ','+line[3]+ ','
                         +line[4]+ ','+line[5]+ ',' +line[6]+ ',' +line[7]+ ',' +line[8]+ ',' +
                         line[9]+ ',' +line[10]+ ','+line[11]+ ','+line[12]+ ','+line[13]+ ','+
                         line[14] + ',' + line[15] + ',' +
                         complete_line)
    else:
        fs.write(main_line+','+complete_line)
f_user.close()
f_ad.close()
f_adv.close()
f_url.close()
f_pos_ad.close()
f_dep_ad.close()
f_gen_ad.close()
f_gen_adv.close()
f_group_ad.close()
f_tr.close()
fs.close()

del d_user
del d_ad
del d_adv
del d_url
del d_pos_ad
del d_dep_ad
del d_gen_ad
del d_gen_adv
del d_group_ad

# Number:7, Description:  Find tf idf output: DL_INPUT_EXTENDED_IDF.txt
Home_Address='/media/fs_Linux_Files/'
f_r = open(Home_Address+'/track2/tokens.txt', "r")
f_r_t = open(Home_Address+'/track2/DL_INPUT_measure.txt', "r")
f_w_t = open(Home_Address+'/track2/DL_INPUT_EXTENDED_IDF.txt', "w")
token_idf = {}
counter=0
for line in f_r:
    line = line.strip().split(',')
    token_idf[int(line[1])] = 0
    counter+=1
    if counter==1070853 :
        break
counter=0
for line in f_r_t:
    if counter%10000000==0:
        print(str(counter))
    line = line.strip().split(',')
    doc=line[7]+'|'+line[8]+'|'+line[9]+'|'+line[10]
    doc_set=set(doc.strip().split('|'))
    for token in doc_set:
        token_idf[int(token)]=int(token_idf[int(token)])+1

for item in token_idf.keys():
    f_w_t.write(str(item)+','+str(token_idf[item])+'\n')

f_w_t.close()
f_r_t.close()
f_r.close()

# Number:8, Description: split train data 1 million, output: T_1.txt
Home_Address='/media/fs_Linux_Files/'
f_r = open(Home_Address+'/track2/DL_INPUT_measure.txt', "r")
f_w = open(Home_Address+'/track2/train/T_1.txt', "w")
# loop over all rows of the CSV file
counter=0
splitter=100000
for line in f_r:
    counter = counter + 1
    f_w.write(line)
    if counter %splitter ==0:
        print(str(counter))
        file_name=Home_Address+'/track2/train/T_'+str(int(counter/splitter)+1)+'.txt'
        f_w.close()
        f_w = open(file_name, "w")
f_w.close()
f_r.close()

# Number:9, Description: Replace tokensids with tokens in PublicTest datasset, output: DL_INPUT_TEST.txt
f_k = open(Home_Address+'/track2/main_data/keyword.txt', 'r')
f_d = open(Home_Address+"/track2/main_data/description.txt", "r")
f_q = open(Home_Address+"/track2/main_data/query.txt", "r")
f_t = open(Home_Address+"/track2/main_data/title.txt", "r")
f_u = open(Home_Address+"/track2/main_data/user.txt", "r")

d_k={}
d_d={}
d_q={}
d_t={}
d_u={}

for line in f_k:
    line = line.strip().split('\t')
    d_k[int(line[0])]=line[1]

for line in f_d:
    line = line.strip().split('\t')
    d_d[int(line[0])]=line[1]

for line in f_q:
    line = line.strip().split('\t')
    d_q[int(line[0])]=line[1]

for line in f_t:
    line = line.strip().split('\t')
    d_t[int(line[0])]=line[1]

for line in f_u:
    line = line.strip().split('\t')
    d_u[int(line[0])]=line[1]+','+line[2]

f_tr = open(Home_Address+"/track2/main_data/TestPublic.txt", "r")
fs = open(Home_Address+"/track2/DL_INPUT_TEST.txt", "w")
def ifnull(var, val):
  if var is None or var == '':
    return val
  return var
counter=0
line = f_tr.readline()
for line in f_tr:
    line_err=line
    counter+=1
    if 0 == counter%1000000:
        print(counter)
    line = line.strip().split(',')
    try:
        fs.write(ifnull(line[1],'0')+','+ifnull(line[2],'1')+','+ifnull(line[12],'-100')+','+ifnull(line[3],'-100')+','+ifnull(line[4],'-100')+','+ifnull(line[5],'-100')+','+ifnull(line[6],'-100')+','+ifnull(d_q[int(line[7])],'-100')+','+ifnull(d_k[int(line[8])],'-100')
                 +','+ifnull(d_t[int(line[9])],'-100')+','+ifnull(d_d[int(line[10])],'-100')+','+ifnull(d_u[int(line[11])],'0,0') +','+ifnull(line[11],'-100')+'\r\n')
    except:
        fs.write(ifnull(line[1],'0')+','+ifnull(line[2],'1')+','+ifnull(line[12],'-100')+','+ifnull(line[3],'-100')+','+ifnull(line[4],'-100')+','+ifnull(line[5],'-100')+','+ifnull(line[6],'-100')+','+ifnull(d_q[int(line[7])],'-100')+','+ifnull(d_k[int(line[8])],'-100')
                 +','+ifnull(d_t[int(line[9])],'-100')+','+ifnull(d_d[int(line[10])],'-100')+',' + '0,0'+','+ifnull(line[11],'-100') + '\r\n')
        #print(str(counter)+','+line[0]+','+line[1]+','+line[2]+','+line[3]+','+line[4]+','+line[5]+','+line[6]+','+line[7]+','+line[8]+','+line[9]+','+line[10]+','+line[11]+'\r\n')

f_d.close()
f_k.close()
f_q.close()
f_t.close()
f_u.close()
f_tr.close()
fs.close()
del d_k
del d_d
del d_q
del d_t
del d_u

# Number:10, Description: Add probabilities and similarity to test data,
# output: DL_INPUT_TEST_measure_user.txt, DL_INPUT_TEST_measure.txt
Home_Address='/media/fs_Linux_Files/'
import Functions_RNN
Functions_RNN.Fill_idf_map()
f_pos_user = open(Home_Address+'/track2/pos_user_CI.txt', "r")
f_dep_user = open(Home_Address+'/track2/dep_user_CI.txt', "r")

f_tr = open(Home_Address+'/track2/DL_INPUT_TEST.txt', "r")
fs = open(Home_Address+'/track2/DL_INPUT_TEST_measure_user.txt', "w")
def ifnull(var, val):
  if var is None or var == '':
    return val
  return var

d_pos_user = {}
d_dep_user = {}

for line in f_pos_user:
    line = line.strip().split(',')
    d_pos_user[int(line[0].split(':')[0]+'00'+line[0].split(':')[1])]=line[1]+':'+line[2]

for line in f_dep_user:
    line = line.strip().split(',')
    d_dep_user[int(line[0].split(':')[0]+'00'+line[0].split(':')[1])]=line[1]+':'+line[2]

#click=0,impression=1,displayurl=2,adid=3,advertiserid=4,depth=5,position=6,queryid=7,
# keywordid=8,titleid=9,description=10,user_gender=11,user_group=12,userid=13,
# pos_user_rate(click:impression)=14,dep_user_rate=15

counter=0
for line in f_tr:
    counter=counter+1
    if 0 == counter%10000000:
        print(counter)
    main_line=line.strip()
    line = line.strip().split(',')
    fs.write(main_line+','+ifnull('0:0' if int(line[6]+'00'+line[13]) not in d_pos_user.keys() else d_pos_user[int(line[6]+'00'+line[13])], '0:0')+','+\
             ifnull('0:0' if int(line[5]+'00'+line[13]) not in d_dep_user.keys() else d_dep_user[int(line[5]+'00'+line[13]) ], '0:0')+'\r\n')

f_pos_user.close()
f_dep_user.close()
f_tr.close()
fs.close()

del d_pos_user
del d_dep_user

Home_Address='/media/fs_Linux_Files/'

import Functions_RNN
Functions_RNN.Fill_idf_map()
f_user = open(Home_Address+'/track2/user_CI.txt', 'r')
f_ad = open(Home_Address+'/track2/ad_CI.txt', "r")
f_adv = open(Home_Address+'/track2/adv_CI.txt', "r")
f_url = open(Home_Address+'/track2/url_CI.txt', "r")

f_pos_ad = open(Home_Address+'/track2/pos_ad_CI.txt', 'r')
f_dep_ad = open(Home_Address+'/track2/dep_ad_CI.txt', "r")
f_gen_ad = open(Home_Address+'/track2/gen_ad_CI.txt', 'r')
f_gen_adv = open(Home_Address+'/track2/gen_adv_CI.txt', "r")
f_group_ad = open(Home_Address+'/track2/group_ad_CI.txt', "r")
f_tr = open(Home_Address+'/track2/DL_INPUT_TEST_measure_user.txt', "r")
fs = open(Home_Address+'/track2/DL_INPUT_TEST_measure.txt', "w")

def ifnull(var, val):
  if var is None or var == '':
    return val
  return var
d_user={}
d_ad={}
d_adv={}
d_url={}

d_pos_ad = {}
d_dep_ad = {}
d_gen_ad = {}
d_gen_adv = {}
d_group_ad = {}


for line in f_user:
    line = line.strip().split(',')
    d_user[line[0]]=line[1]+':'+line[2]

for line in f_ad:
    line = line.strip().split(',')
    d_ad[line[0]]=line[1]+':'+line[2]

for line in f_adv:
    line = line.strip().split(',')
    d_adv[line[0]]=line[1]+':'+line[2]

for line in f_url:
    line = line.strip().split(',')
    d_url[line[0]]=line[1]+':'+line[2]

for line in f_pos_ad:
    line = line.strip().split(',')
    d_pos_ad[line[0]]=line[1]+':'+line[2]

for line in f_dep_ad:
    line = line.strip().split(',')
    d_dep_ad[line[0]]=line[1]+':'+line[2]

for line in f_gen_ad:
    line = line.strip().split(',')
    d_gen_ad[line[0]]=line[1]+':'+line[2]

for line in f_gen_adv:
    line = line.strip().split(',')
    d_gen_adv[line[0]]=line[1]+':'+line[2]

for line in f_group_ad:
    line = line.strip().split(',')
    d_group_ad[line[0]]=line[1]+':'+line[2]

#click=0,impression=1,displayurl=2,adid=3,advertiserid=4,depth=5,position=6,queryid=7,
# keywordid=8,titleid=9,description=10,user_gender=11,user_group=12,userid=13,
# pos_user_rate(click:impression)=14,dep_user_rate=15,
# user_rate=16,ad_rate=17,adv_rate=18,url_rate=19,pos_ad_rate=20,dep_ad_rate=21
# gen_ad_rate=22,gen_adv_rate=23,group_ad_rate=24
#tfidf(q7-k8)=25,tfidf(q7-t9)=26,tfidf(q7-d10)=27,tfidf(k8-t9)=28,tfidf(k8-d10)=29,tfidf(t9-d10)=30
#sim(q7-k8)=31,sim(q7-t9)=32,sim(q7-d10)=33,sim(k8-t9)=34,sim(k8-d10)=35,sim(t9-d10)=36
counter=0
for line in f_tr:
    counter=counter+1
    if 0 == counter%1000000:
        print(counter)
    main_line=line.strip()
    line = line.strip().split(',')

    complete_line = ifnull('0:0' if line[13] not in d_user.keys() else d_user[line[13]], '0:0') + ','\
                    + ifnull('0:0' if line[3] not in d_ad.keys() else d_ad[line[3]], '0:0') + ',' \
                    + ifnull('0:0' if line[4] not in d_adv.keys() else d_adv[line[4]], '0:0') + ',' \
                    + ifnull('0:0' if line[2] not in d_url.keys() else d_url[line[2]], '0:0') + ',' \
                    + ifnull('0:0' if line[6] + ':' + line[3] not in d_pos_ad else d_pos_ad[line[6] + ':' + line[3]], '0:0') + ',' \
                    + ifnull('0:0' if line[5] + ':' + line[3] not in d_dep_ad else d_dep_ad[line[5] + ':' + line[3]], '0:0') + ',' \
                    + ifnull('0:0' if line[11] + ':' + line[3] not in d_gen_ad else d_gen_ad[line[11] + ':' + line[3]],'0:0') + ',' \
                    + ifnull('0:0' if line[11] + ':' + line[4] not in d_gen_adv else d_gen_adv[line[11] + ':' + line[4]], '0:0') + ',' \
                    + ifnull('0:0' if line[12] + ':' + line[3] not in d_group_ad else d_group_ad[line[12] + ':' + line[3]], '0:0') + ','\
                    + str(Functions_RNN.Compute_cosine_sim_TFIDF(line[7], line[8])) + ',' \
                    + str(Functions_RNN.Compute_cosine_sim_TFIDF(line[7], line[9])) + ',' \
                    + str(Functions_RNN.Compute_cosine_sim_TFIDF(line[7], line[10])) + ',' \
                    + str(Functions_RNN.Compute_cosine_sim_TFIDF(line[8], line[9])) + ',' \
                    + str(Functions_RNN.Compute_cosine_sim_TFIDF(line[8], line[10])) + ',' \
                    + str(Functions_RNN.Compute_cosine_sim_TFIDF(line[9], line[10])) + ',' \
                    + str(Functions_RNN.Compute_cosine_sim(line[7], line[8])) + ',' \
                    + str(Functions_RNN.Compute_cosine_sim(line[7], line[9])) + ',' \
                    + str(Functions_RNN.Compute_cosine_sim(line[7], line[10])) + ',' \
                    + str(Functions_RNN.Compute_cosine_sim(line[8], line[9])) + ',' \
                    + str(Functions_RNN.Compute_cosine_sim(line[8], line[10])) + ',' \
                    + str(Functions_RNN.Compute_cosine_sim(line[9], line[10])) \
                    + '\r\n'
    if int(line[0]) > 1:
        for i in range(int(line[0])):
            if i == int(line[0]) - 1:
                fs.write('1' + ',' + str(int(line[1]) - int(line[0]) + 1) + ',' + line[2] + ',' + line[3] + ','
                         + line[4] + ',' + line[5] + ',' + line[6] + ',' + line[7] + ',' + line[8] + ',' +
                         line[9] + ',' + line[10] + ',' + line[11] + ',' + line[12] + ',' + line[13] + ',' +
                         line[14] + ',' + line[15] + ',' +
                         complete_line)
            else:
                fs.write('1' + ',' + '1' + ',' + line[2] + ',' + line[3] + ','
                         + line[4] + ',' + line[5] + ',' + line[6] + ',' + line[7] + ',' + line[8] + ',' +
                         line[9] + ',' + line[10] + ',' + line[11] + ',' + line[12] + ',' + line[13] + ',' +
                         line[14] + ',' + line[15] + ',' +
                         complete_line)
    else:
        fs.write(main_line + ',' + complete_line)


f_user.close()
f_ad.close()
f_adv.close()
f_url.close()
f_tr.close()
fs.close()
f_pos_ad.close()
f_dep_ad.close()
f_gen_ad.close()
f_gen_adv.close()
f_group_ad.close()

del d_user
del d_ad
del d_adv
del d_url
del d_pos_ad
del d_dep_ad
del d_gen_ad
del d_gen_adv
del d_group_ad

# Number:11, Description: split Test data to validation and test, output: V_1.txt, TEST.txt
Home_Address='/media/fs_Linux_Files/'
f_r = open(Home_Address+'/track2/DL_INPUT_TEST_measure.txt', "r")
f_w = open(Home_Address+'/track2/train/V_1.txt', "w")
f_w_test = open(Home_Address+'/track2/train/TEST.txt', "w")
# loop over all rows of the CSV file
splitter=100000
Total_validation_record=4000000
counter=0
for line in f_r:
    if counter <= Total_validation_record:
        counter = counter + 1
        f_w.write(line)
        if counter %splitter ==0:
            print(str(counter))
            file_name=Home_Address+'/track2/train/V_'+str(int(counter/splitter)+1)+'.txt'
            f_w.close()
            f_w = open(file_name, "w")
    else:
        counter = counter + 1
        f_w_test.write(line)

f_w.close()
f_r.close()
f_w_test.close()



