import torch
import numpy as np

def rmv_ent(db, n):
    b = torch.cat([db[:n], db[n+1:]])
    return b

def rmv_db(db):
    tmp = []
    for i in range(len(db)):
        tmp.append(rmv_ent(db, i))
    return tmp

def create_db_and_parallels(n_entries):
    db = torch.rand(n_entries) > 0.5
    pds = rmv_db(db)
    return db, pds

# db10, pds10 = create_db_and_parallels(10)
# print(db10)

db_100, pdbs_100 = create_db_and_parallels(100)
def query_mean(db):
    return db.float().mean()

def cal_sensitivity(query, n):
    db, pdbs = create_db_and_parallels(n)
    full_db_result = query(db)

    sensitivity = 0
    for pdb in pdbs:
        pdb_result = query(pdb)
        db_distance = torch.abs(pdb_result - full_db_result)

        if (db_distance > sensitivity):
            sensitivity = db_distance

    return sensitivity

def query(db, noise):
    true_result = torch.mean(db.float()) # 원래 고유 db의  값
    first_coin_flip = (torch.rand(len(db)) > noise).float() # noise 주는 값 기준으로, noise 값 이상일 때 1 도출 (의도적noise)
    second_coin_flip = (torch.rand(len(db))> 0.5).float() # 동전던지기 균일 0.5 확률 2번째 동전
    augmented_database = db.float() * first_coin_flip + (1-first_coin_flip) * second_coin_flip #확장 데이터셋
    sk_result = augmented_database.float().mean() #확장 데이터셋의 값은 skewed 되어 있겠지
    private_result = ((sk_result / noise) - 0.5) * noise / (1-noise) # 위의 skewed 값을 제자리 잡아주는 것.
    return private_result, true_result

db,pdbs = create_db_and_parallels(1000)
private_result , true_result = query(db, noise = 0.1)
print("With Noise: " + str(private_result))
print("Withoud Noise: " + str(true_result))
