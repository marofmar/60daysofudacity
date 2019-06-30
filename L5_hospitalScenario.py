import numpy as np

num_teachers = 10
num_examples = 10000
num_labels = 10

preds = (np.random.rand(num_teachers, num_examples) * num_labels).astype(int).transpose(1,0)
#print(len(preds))
new_labels = list()
for an_image in preds:
    label_counts = np.bincount(an_image, minlength = num_labels)
    epsilon = 0.1
    beta = 1/ epsilon

    for i in range(len(label_counts)):
        label_counts[i] += np.random.laplace(0, beta, 1)

    new_label = np.argmax(label_counts)

    new_labels.append(new_label)

print(len(new_labels))

# an_image = preds[:,0]
# # print(an_image)
# # print(np.bincount(an_image, minlength = num_labels))
# label_counts = np.bincount(an_image, minlength = num_labels)
#
# # DP
# epsilon = 0.1
# beta = 1/epsilon
#
# for i in range(len(label_counts)):
#     label_counts[i] += np.random.laplace(0, beta, 1)
#
# print(label_counts)
#
# print(np.argmax(label_counts))
