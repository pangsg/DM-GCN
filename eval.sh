#!/bin/bash
# Restaurants
python eval.py --dataset Restaurants --hidden_dim 300 --rnn_hidden 300 \
    --top_k 2 --head_num 3 --batch_size 32 --save_dir ./saved_models/best_model_rest.pt
echo -e "\n"

# Tweets
python eval.py --dataset Tweets --hidden_dim 300 --rnn_hidden 300 \
    --top_k 4 --head_num 3 --batch_size 32 --save_dir ./saved_models/best_model_tweet.pt
echo -e "\n"

# Laptops
python eval.py --dataset Laptops --hidden_dim 204 --rnn_hidden 102 \
    --top_k 2 --head_num 3 --batch_size 8 --save_dir ./saved_models/best_model_lap.pt
echo -e "finish"
