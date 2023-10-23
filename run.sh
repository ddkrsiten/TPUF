nohup python -u main.py --dataset douban_book --maxlen 50 --model TPUF --s_model sas --l2_emb 0.3 --combine_weight 0.5 --hidden_units 32 --num_epochs 500 --gpu 1 --loss_weight 1.1 > logs/aa.txt 2>&1 &
nohup python -u main.py --dataset Amazon_book --maxlen 50 --model TPUF --s_model sas --l2_emb 0.6 --combine_weight 0.5 --hidden_units 32 --num_epochs 300 --gpu 2 --loss_weight 0.9 > logs/bb.txt 2>&1 &
nohup python -u main.py --dataset Amazon_cloth --maxlen 50 --model TPUF --s_model sas --l2_emb 0.4 --combine_weight 0.4 --hidden_units 32 --num_epochs 300 --gpu 3 --loss_weight 0.8 > logs/cc.txt 2>&1 &



nohup python -u main.py --dataset douban_book --maxlen 50 --model TPUF --s_model mf --l2_emb 0.4 --combine_weight 0.5 --hidden_units 32 --num_epochs 500 --gpu 4 --loss_weight 1.0 > logs/dd.txt 2>&1 &
nohup python -u main.py --dataset Amazon_book --maxlen 50 --model TPUF --s_model mf --l2_emb 0.5 --combine_weight 0.5 --hidden_units 32 --num_epochs 300 --gpu 5 --loss_weight 0.8 > logs/ee.txt 2>&1 &
nohup python -u main.py --dataset Amazon_cloth --maxlen 50 --model TPUF --s_model mf --l2_emb 0.4 --combine_weight 0.4 --hidden_units 32 --num_epochs 300 --gpu 6 --loss_weight 0.8 > logs/ff.txt 2>&1 &


nohup python -u main.py --dataset douban_book --maxlen 50 --model TPUF --s_model mlp --l2_emb 0.4 --combine_weight 0.5 --hidden_units 32 --num_epochs 500 --gpu 4 --loss_weight 1.0 > logs/gg.txt 2>&1 &
nohup python -u main.py --dataset Amazon_book --maxlen 50 --model TPUF --s_model mlp --l2_emb 0.6 --combine_weight 0.5 --hidden_units 32 --num_epochs 300 --gpu 5 --loss_weight 0.9 > logs/hh.txt 2>&1 &
nohup python -u main.py --dataset Amazon_cloth --maxlen 50 --model TPUF --s_model mlp --l2_emb 0.4 --combine_weight 0.4 --hidden_units 32 --num_epochs 300 --gpu 6 --loss_weight 0.8 > logs/ii.txt 2>&1 &