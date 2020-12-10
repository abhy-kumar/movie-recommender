import numpy as np
from lightfm import LightFM
from lightfm.datasets import fetch_movielens

data = fetch_movielens()

print(repr(data["train"]))
print(repr(data["test"]))

model = LightFM(loss="warp")
model.fit(data["train"], epochs=50, num_threads=3)


def rec_init(model, data, user_id):
    n_users, n_items = data["train"].shape
    for user in user_id:
        known_positive = data["item_labels"][data["train"].tocsr()[user].indices]
        scores = model.predict(user, np.arrange(n_items))
        top_items = data["item_labels"][np.argsort(scores)]
