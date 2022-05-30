import torch
import os
import time
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from dataset import ParagraphDataset
from model import EGNN_Model


def get_dataloader(pdb_H_L_csv, pdb_folder_path, batch_size=1):
    '''
    '''
    ds = ParagraphDataset(pdb_H_L_csv, pdb_folder_path)
    
    return DataLoader(dataset=ds, batch_size=batch_size)


def evaluate_model(model, dataloader, device):
    '''
    Evaluate trained model against new dataset
    '''
    
    # loop over batches in test set
    for _, ((feats, coors, edges), (pdb_code, AAs, AtomNum, chain, IMGT, x, y, z)) in enumerate(dataloader):

        feats = feats.to(device)
        coors = coors.to(device)
        edges = edges.to(device)
        
        # make our predictions with our saved model
        pred = model.forward(feats, coors, edges)
        pred = pred.to(device)
        
        # create detailed record to allow easily analysis
        num_atoms = len(AAs)
        num_complexes = len(pdb_code)
        for i in range(num_atoms):
            for j in range(num_complexes):
                row = [pdb_code[j],
                       AAs[i][j],
                       AtomNum[i][j],
                       chain[i][j],
                       IMGT[i][j],
                       x[i][j],
                       y[i][j],
                       z[i][j],
                       torch.sigmoid(pred)[j][i][0].item()]
                try:
                    detailed_record = np.append(detailed_record, [row], axis=0)
                except UnboundLocalError:
                    detailed_record = np.array([row], dtype=object)
                    
    # convert ndarray to df for easier viewing
    detailed_record_df = pd.DataFrame(detailed_record,
                                      columns=["pdb", "AA", "Atom_Num", "Chain", "IMGT", "x", "y", "z", "pred"])

    # remove missing residues from results
    detailed_record_df = detailed_record_df[detailed_record_df["AA"] != ""].reset_index(drop=True)
    
    return detailed_record_df
