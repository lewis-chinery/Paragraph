import argparse
import os
import sys
import torch
import time
from Paragraph.model import EGNN_Model
from Paragraph.predict import get_dataloader, evaluate_model


description='''
PARAGRAPH

Paratope prediction using Equivariant Graph Neural Networks
Predicitons take approximately 0.1s per structure

Requirements:
 - torch, scipy, einops, pandas, numpy
 
e.g.
 - Paragraph --pdb_H_L_csv     /path/to/key.csv
             --pdb_folder_path /path/to/pdb/files/
             --out_path        /path/to/saved/predictions.csv

'''

epilogue='''
Author: Lewis Chinery (lewis.chinery@jesus.ox.ac.uk)
        Charlotte M. Deane (deane@stats.ox.ac.uk)
Contact: opig@stats.ox.ac.uk
 
'''

parser = argparse.ArgumentParser(prog="Paragraph", description=description, epilog=epilogue,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-e", "--example", help="run Paragraph using default weights on example data", action='store_true', default=False)
parser.add_argument("-k", "--pdb_H_L_csv", help="abspath to csv key containing names of pdb files \
                     (without .pdb), and H & L chain IDs", default=None)
parser.add_argument("-i", "--pdb_folder_path", help="abspath to folder containing pdb files", default=None)
parser.add_argument("-o", "--out_path", help="abspath to csv file where predictions will be saved", default=None)
parser.add_argument("-w", "--weights", help="abspath to weights to be used by network (default is pre-trained)", default=None)
args = parser.parse_args()


def main():
    
    # show help menu if no options given
    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(0)
        
    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # args
    example = args.example
    pdb_H_L_csv = args.pdb_H_L_csv
    pdb_folder_path = args.pdb_folder_path
    out_path = args.out_path
    saved_model_path = args.weights
    
    # paths used in example
    root_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-5])
    src_dir = os.path.join(root_dir, "Paragraph")
    example_dir = os.path.join(root_dir, "example")
    trained_model_path = os.path.join(root_dir, "trained_model")
    
    # use pre-trained weights if no additonal ones given
    saved_model_path = os.path.join(trained_model_path, "pretrained_weights.pt") if args.weights is None else args.weights
    
    # run example
    if example and len(sys.argv) > 2:
        raise ValueError("No other arguments can be provided when the example is run")
    elif example:
        pdb_H_L_csv = os.path.join(example_dir, "pdb_H_L_key.csv")
        pdb_folder_path = os.path.join(example_dir, "pdbs")
        predictions_output_path = os.path.join(example_dir, "example_predictions.csv")
    # ensure input is OK when not running example
    elif pdb_H_L_csv is None or pdb_folder_path is None or out_path is None:
        raise ValueError("pdb_H_L_csv, pdb_folder_path, and out_path should all be provided when running on custom data")
    else:
        pdb_H_L_csv = pdb_H_L_csv + ".csv" if pdb_H_L_csv[-4:] != ".csv" else pdb_H_L_csv
        predictions_output_path = out_path + ".csv" if out_path[-4:] != ".csv" else out_path
        
    # network architecture
    num_feats = 22  # 20D one-hot encoding of AA type and 2D one-hot encoding of chain ID
    graph_hidden_layer_output_dims = [num_feats]*6
    linear_hidden_layer_output_dims = [10]*2
    saved_net = EGNN_Model(num_feats = num_feats,
                           graph_hidden_layer_output_dims = graph_hidden_layer_output_dims,
                           linear_hidden_layer_output_dims = linear_hidden_layer_output_dims)
    
    # load weights
    try:
        saved_net.load_state_dict(torch.load(saved_model_path))
    except RuntimeError:
        saved_net.load_state_dict(torch.load(saved_model_path, map_location=torch.device('cpu')))
    saved_net = saved_net.to(device)
    
    # dataloader    
    dl = get_dataloader(pdb_H_L_csv, pdb_folder_path)
    
    # predict
    print("\nEvaluating using weight file:\n - {}\n".format(saved_model_path))#.split("Paragraph")[-1]))
    start_time = time.time()
    detailed_record_df = evaluate_model(model = saved_net,dataloader = dl, device = device)
    
    # save results
    detailed_record_df.to_csv(predictions_output_path, index=False)
    print("Results saved to:\n - {}\n".format(predictions_output_path))#.split("Paragraph")[-1]))
    print("Total evaluation time:\n - {:.3f}s\n".format(time.time()-start_time))

    
if __name__ == '__main__':
    main()
    