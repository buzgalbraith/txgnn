from txgnn import TxData, TxGNN, TxEval
import argparse


if __name__ == '__main__':
  ## if want to use bash args 
  use_manual_args = False
  if not use_manual_args:
      print("reading in arguments:")
      parser = argparse.ArgumentParser(description='Run the evaluation pipeline')
      parser.add_argument('--finetune_on_head', type=str, default=None, help='The method to use for embeddings')
      parser.add_argument('--finetune_on_relation', type=str, default=None, help='The method to use for embeddings')
      parser.add_argument('--finetune_on_tail', type=str, default=None, help='The method to use for embeddings')
      parser.add_argument('--pretrain', type=bool, default=True, help='Whether to pretrain the model')
      parser.add_argument('--finetune', type=bool, default=True, help='Whether to finetune the model')
      parser.add_argument('--save_path', type=str, default='saved_models/txgnn.pt', help='The path to save the model')
      parser.add_argument('--finetune_result_path', type=str, default='logs/fine_tuning', help='The path to save the fine tuning results')
      args = parser.parse_args()
      if args.finetune_on_head is None or args.finetune_on_relation is None or args.finetune_on_tail is None:
         finetune_on = None
      else:
        finetune_on = (args.finetune_on_head, args.finetune_on_relation, args.finetune_on_tail)
      pretrain = args.pretrain
      finetune = args.finetune
      save_path = args.save_path
      finetune_result_path = args.finetune_result_path
  else:
      finetune_on = ('drug', 'drug_protein', 'gene/protein')
      save_path = 'saved_models/txgnn.pt'
      pretrain = True
      finetune = True
      finetune_result_path = 'logs/fine_tuning'
  print(finetune_on, pretrain, finetune, save_path, finetune_result_path)
  # make data object 
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
  ## save model just in case 
  TxGNN.save_model(path = save_path)
  if finetune:
    TxGNN.finetune(n_epoch = 500, 
                learning_rate = 5e-4,
                train_print_per_n = 5,
                valid_per_n = 20,
                 finetune_on = finetune_on,
                # finetune_on= ('drug', 'contraindication', 'disease'),
                save_name = finetune_result_path)
  TxGNN.save_model(path = save_path)