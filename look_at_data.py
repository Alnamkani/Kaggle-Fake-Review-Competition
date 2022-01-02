import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# data_tr = pd.read_csv('reviews_train.csv',quotechar='"',usecols=[0,1,2,3],dtype={'real review?': int,'category': str, 'rating': int, 'text_': str})
# data_val = pd.read_csv('reviews_validation.csv',quotechar='"',usecols=[0,1,2,3],dtype={'real review?': int,'category': str, 'rating': int, 'text_': str})
# per = len(data_val) / (len(data_tr) + len(data_val))
# all_data = pd.concat([data_val, data_tr])

# # data = all_data.sample(frac=per)

# all_data['len_str'] = [len(all_data.iloc[x]['text_']) for x in range(len(all_data))]
# all_data['len_str'] /= max(all_data['len_str'])
# max_len = 0
# for i in range(len(all_data)):
#     temp_len = len(all_data.iloc[i]['text_'])
#     all_data.iloc[i]['len_str'] = temp_len
#     if max_len < temp_len:
#         max_len = temp_len
# all_data['len_str'] /= max_len

# print(all_data.describe())

# data = pd.read_csv('reviews_test_attributes.csv',quotechar='"',usecols=[0,1,2,3],dtype={'real review?': int,'category': str, 'rating': int, 'text_': str})

# for cat in data_tr['category'].unique():
#     m = data.groupby('category').get_group(cat)['real review?'].mean()
#     print(f"cat: {cat}, mean = {m}")
#     # print(len(data_tr.groupby('category').get_group(cat)))

 
# for cat in data['rating'].unique():
#     m = data.groupby('rating').get_group((cat))['real review?'].mean()
#     print(f"rating: {cat}, mean = {m}")


# for rat in data['rating'].unique():
#     m = data.groupby('rating').get_group(rat)
#     print(f"{rat}: {len(m)}")

# data = data_tr['rating'] / 5
# print(data.head())
# num_fake = 0
# num_real = 0
# length_fake = []
# length_real = []
# num_upper_fake = 0
# num_upper_real = 0
# num_and_real = 0
# num_but_real = 0
# num_and_fake = 0
# num_but_fake = 0
# for i in range(len(all_data)):
#     review = all_data.iloc[i]['real review?']
#     s = all_data.iloc[i]['text_']
#     numbers_and = s.count('and')
#     numbers_but = s.count('but')
#     cap = 0
#     for j in range(len(s)):
#         if not s[j].isalpha():
#             cap += 1
#     if review == 0:
#         num_fake += 1
#         # length_fake.append(len(s))
#         num_upper_fake += cap / len(s)
#         num_but_fake += numbers_but
#         num_and_fake += numbers_and
#     else:
#         num_real += 1
#         # length_real.append(len(s))
#         num_upper_real += cap / len(s)
#         num_but_real += numbers_but
#         num_and_real += numbers_and


# print(num_upper_real / num_real)
# # print(num_but_real )
# print(num_real)
# print("$$$$$$$$$$$")
# print(num_upper_fake / num_fake)
# # print(num_but_fake)
# print(num_fake)

# print(np.percentile(length_fake, [0, 25, 50, 75, 100]))

# print(np.percentile(length_real, [0, 25, 50, 75, 100]))



data_test = pd.read_csv('reviews_test_attributes.csv',quotechar='"',usecols=[0,1,2,3],dtype={'real review?': int,'category': str, 'rating': int, 'text_': str})

print(data_test.describe())