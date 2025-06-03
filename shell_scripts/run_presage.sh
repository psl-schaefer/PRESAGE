ds=replogle_k562_essential_unfiltered
seed=seed_0
python3 src/train_presage.py --config=./configs/singles_config.json --data_config=./configs/${ds}_config.json --data.seed=./splits/${ds}_random_splits/${seed}.json --model.pathway_files=./sample_files/prior_files/sample.onlgo.txt  --data.dataset=${ds} --training.predictions_file=./intermediate_data/model_predictions/predictions.PRESAGE.${ds}.${seed}.csv.gz --training.embedding_file=./intermediate_data/model_embeddings/${ds}.${seed} --training.attention_file=./intermediate_data/model_attention/${ds}.${seed} --training.eval_test=True 
