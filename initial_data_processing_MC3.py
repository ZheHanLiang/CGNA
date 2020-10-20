##########################################################################
# @File: initial_data_processing_MC3.py
# @Author: Zhehan Liang
# @Date: 7/21/2020
# @Intro: Adjust the original data, process the number of nodes, the minimum degree of the deleted edge node,
# the data format, etc., and delete a certain number of non-duplicated edges respectively,
# and finally save them as two new graphs of the source domain and the target domain, and save the format As .txt
# tips: Real-world network, e.g. MC3-MC3
# @Data source: http://vacommunity.org/VAST+Challenge+2018+MC3
##########################################################################

import numpy as np
import os
import time
import argparse
import csv

def data_processing():
    ## Parameter setting
    dataset = "MC3"
    input_dir = "./data/graph_data/MC3/" # Original data file path
    output_dir = "./data/graph_edge/" # New data file storage path

    ## Path setting
    input_data_dir_call = input_dir + "calls.csv" # Original data path for calls
    input_data_dir_email = input_dir + "emails.csv" # Original data path for emails

    ## Determine the node, only remain nodes whose contact times greater than or equal to num_1
    num_1 = 400
    weeks = 10000
    seconds = weeks * 24 * 3600 + 5000
    user_call = {}
    user_email = {}
    with open(input_data_dir_call, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            a = row[0] # Caller
            b = row[2] # Reciever
            time = row[3]
            if int(time)<seconds:
                if a not in user_call:
                    user_call[a] = 1
                else:
                    user_call[a] += 1
                if b not in user_call:
                    user_call[b] = 1
                else:
                    user_call[b] += 1
            else: break
    with open(input_data_dir_email, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            a = row[0] # Sender
            b = row[2] # Reciever
            time = row[3]
            if int(time)<seconds:
                if a not in user_email:
                    user_email[a] = 1
                else:
                    user_email[a] += 1
                if b not in user_email:
                    user_email[b] = 1
                else:
                    user_email[b] += 1
            else: break
    print("User number:", len(user_call), len(user_email))
    del_call = []
    del_email = []
    # Delete 'call' which don't meet the criteria
    for key in user_call:
        if user_call[key]<num_1:
            del_call.append(key)
    for user in del_call:
        del user_call[user]
    # Delete 'email' which don't meet the criteria
    for key in user_email:
        if user_email[key]<num_1:
            del_email.append(key)
    for user in del_email:
        del user_email[user]
    users_set = user_call.keys() & user_email.keys()
    print("Common user number:", len(users_set))
    users_dict = {}
    index = 0
    for user in users_set:
        users_dict[user] = index
        index += 1
    ## Determine the edges, only remain the edges whose contact times greater than or equal to num_2
    num_2 = 1
    user_call_edges = {}
    user_email_edges = {}
    with open(input_data_dir_call, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            a = row[0]
            b = row[2]
            time = row[3]
            if int(time)<seconds:
                if a not in users_set or b not in users_set:
                    continue
                if a+'-'+b in user_call_edges:
                    user_call_edges[a+'-'+b] += 1
                elif b+'-'+a in user_call_edges:
                    user_call_edges[b+'-'+a] += 1
                else:
                    user_call_edges[a+'-'+b] = 1
            else: break
    with open(input_data_dir_email, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            a = row[0]
            b = row[2]
            time = row[3]
            if int(time)<seconds:
                if a not in users_set or b not in users_set:
                    continue
                if a+'-'+b in user_email_edges:
                    user_email_edges[a+'-'+b] += 1
                elif b+'-'+a in user_email_edges:
                    user_email_edges[b+'-'+a] += 1
                else:
                    user_email_edges[a+'-'+b] = 1
            else: break
    print("Edge number:", len(user_call_edges), len(user_email_edges))
    # Delete edges whose contact times less than num_2
    del_call = []
    del_email = []
    # Delete 'call' which don't meet the criteria
    for key in user_call_edges:
        if user_call_edges[key]<num_2:
            del_call.append(key)
    for edge in del_call:
        del user_call_edges[edge]
    # Delete 'email' which don't meet the criteria
    for key in user_email_edges:
        if user_email_edges[key]<num_2:
            del_email.append(key)
    for edge in del_email:
        del user_email_edges[edge]
    print("Discarded edge number:", len(user_call_edges), len(user_email_edges))
    ## Save edges
    output_source_dir = output_dir + "{0}_source_edges.txt".format(dataset) # Source graph path
    output_target_dir = output_dir + "{0}_target_edges.txt".format(dataset) # Target graph path
    edges_call = ""
    edges_email = ""
    users_set_call = set()
    users_set_email = set()
    for edge in user_call_edges.keys():
        users = edge.split('-')
        users_set_call.add(users[0])
        users_set_call.add(users[1])
        edges_call += "{0} {1}\n".format(users_dict[users[0]], users_dict[users[1]])
    if len(users_set_call)<len(users_set):
        for user in users_set:
            if user not in users_set_call:
                edges_call += "{0} {0}\n".format(users_dict[user])
    if not os.path.exists(output_dir): # Check whether the output path exists, if not, create it
        os.makedirs(output_dir)
    f = open(output_source_dir, 'w', encoding='utf-8')
    f.write(edges_call)
    f.close()
    for key in user_email_edges.keys():
        users = key.split('-')
        users_set_email.add(users[0])
        users_set_email.add(users[1])
        edges_email += "{0} {1}\n".format(users_dict[users[0]], users_dict[users[1]])
    if len(users_set_email)<len(users_set):
        for user in users_set:
            if user not in users_set_email:
                edges_email += "{0} {0}\n".format(users_dict[user])
    f = open(output_target_dir, 'w', encoding='utf-8')
    f.write(edges_email)
    f.close()
    print("Real user number:", len(users_set_call), len(users_set_email))

if __name__ == "__main__":
    data_processing()