from txgnn import TxData, TxGNN, TxEval

## run parameters
save_path = 'saved_models/txgnn.pt'
pretrain = False
finetune = True
finetune_result_path = 'logs/fine_tuning'
  

if __name__ == '__main__':
  ## make data object 
  TxData = TxData(data_folder_path = 'data/dataverse/')
  TxData.prepare_split(split = 'complex_disease', seed = 42)
  TxGNN = TxGNN(data = TxData, 
                weight_bias_track = False,
                proj_name = 'TxGNN', # wandb project name
                exp_name = 'TxGNN', # wandb experiment name
              # device = 'cuda:0' # define your cuda device
              device = 'cpu'
                )
  TxGNN.model_initialize(n_hid = 100, # number of hidden dimensions
                        n_inp = 100, # number of input dimensions
                        n_out = 100, # number of output dimensions
                        proto = True, # whether to use metric learning module
                        proto_num = 3, # number of similar diseases to retrieve for augmentation
                        attention = False, # use attention layer (if use graph XAI, we turn this to false)
                        sim_measure = 'all_nodes_profile', # disease signature, choose from ['all_nodes_profile', 'protein_profile', 'protein_random_walk']
                        agg_measure = 'rarity', # how to aggregate sim disease emb with target disease emb, choose from ['rarity', 'avg']
                        num_walks = 200, # for protein_random_walk sim_measure, define number of sampled walks
                        path_length = 2 # for protein_random_walk sim_measure, define path length
                        )
  ## pre train
  if pretrain:               
    TxGNN.pretrain(n_epoch = 2, 
                 learning_rate = 1e-3,
                 batch_size = 1024, 
                 train_print_per_n = 20)
  else:
      TxGNN.load_pretrained(save_path)
  if finetune:
    TxGNN.finetune(n_epoch = 500, 
                learning_rate = 5e-4,
                train_print_per_n = 5,
                valid_per_n = 20,
                 finetune_on = ('drug', 'drug_protein', 'gene/protein'),
                # finetune_on= ('drug', 'contraindication', 'disease'),
                save_name = finetune_result_path)
  TxGNN.save_model(path = save_path)