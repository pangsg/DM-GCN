#!/bin/bash
# Restaurants
echo -e "\n evaluating Restaurant with glove"
python eval.py --dataset Restaurants --hidden_dim 300 --rnn_hidden 300 --emb_type glove\
    --top_k 2 --head_num 3 --batch_size 32 --save_dir ./saved_models/best_model_rest.pt

# Tweets
echo -e "\n evaluating Twitter with glove"
python eval.py --dataset Tweets --hidden_dim 300 --rnn_hidden 300 --emb_type glove\
    --top_k 4 --head_num 3 --batch_size 32 --save_dir ./saved_models/best_model_tweet.pt

# Laptops
echo -e "\n evaluating Laptop with glove"
python eval.py --dataset Laptops --hidden_dim 204 --rnn_hidden 102 --emb_type glove\
    --top_k 2 --head_num 3 --batch_size 8 --save_dir ./saved_models/best_model_lap.pt

# Restaurants
echo -e "\n evaluating Restaurant with bert"
python eval.py --dataset Restaurants --hidden_dim 300 --rnn_hidden 300 --emb_type bert\
    --top_k 2 --head_num 3 --batch_size 32 --save_dir ./saved_models/best_model_rest_bert.pt

# Twitters
echo -e "\n evaluating Twitter with bert"
python eval.py --dataset Laptops --hidden_dim 300 --rnn_hidden 300 --emb_type bert\
    --top_k 2 --head_num 3 --batch_size 32 --save_dir ./saved_models/best_model_tweet_bert.pt

# Laptops
echo -e "\n evaluating Laptop with bert"
python eval.py --dataset Laptops --hidden_dim 204 --rnn_hidden 102 --emb_type bert\
    --top_k 2 --head_num 3 --batch_size 8 --save_dir ./saved_models/best_model_lap_bert.pt
echo -e "finish"
