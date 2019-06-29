# set parameters
epsilon = 0.5

# load dependencies
import numpy as np
import torch

# create database with 0 and 1
"""
rmv_ent(db,n): remove Nth entry in a dataset
rmv_db(db): iterate "rmv_ent" function on every dataset

"""
def rmv_ent(db, n):
    b = torch.cat([db[:n], db[n+1:]])
    return b

def rmv_db(db):
    tmp = []
    for i in range(len(db)):
        tmp.append(rmv_ent(db, i))
    return tmp

def creat_db_and_paralleles(n_entries):
    db = torch.rand(n_entries) > 0.5
    pds = rmv_db(db)

    return db, pds

### make databaase for this project
db, pdbs = creat_db_and_paralleles(100)

def sum_query(db):
    return db.sum()
'''
beta: spread parameter
'''

def laplacian_mechanism(db, query, sensitivity):
    beta = sensitivity / epsilon
    noise = (torch.tensor(np.random.laplace(0, beta, 1))).float()
    return (query(db) + noise)

print(sum_query(db))
print(laplacian_mechanism(db, sum_query, 1))
