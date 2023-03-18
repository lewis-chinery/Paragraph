import os
import pandas as pd


def breakdown_line(line):
    '''
    Get the chain ID, IMGT number, and B factor from a line in a pdb

    :param line: str line of pdb file
    :returns: 3 strs - chain ID, IMGT number, B factor
    '''

    chain, imgt, temp = None, None, None
    if line[:4] == "ATOM":
        chain = line[21]
        imgt = line[23:26].strip()
        temp = line[60:66]
    
    return chain, imgt, temp


def get_preds_as_str_2_decimal_places(df_pred):
    '''
    '''
    df_pred["pred_2dp"] = df_pred["pred"]*100
    df_pred["pred_2dp"] = df_pred["pred_2dp"].round(2)
    df_pred["pred_2dp"] = df_pred["pred_2dp"].apply(lambda pred: "{:.2f}".format(pred))
    return df_pred


def replace_B_factors_with_paratope_predictions(pdb_file, predictions_csv, inplace=False, new_suffix="_Paragraph"):
    '''
    This makes visualising Paragraph's predictions in PyMOL easier

    :param pdb_file: abs path to pdb file
    :param predictions_csv: abs path to Paragraph's output predictions
    :param inplace: bool overwrite existing pdb file if True
    :param new_suffix: str suffix to apply to new pdb file if not updating inplace
    '''

    new_pdb_file = pdb_file[:-4] + f"{new_suffix}.pdb"

    # get predictions for pdb of interest only
    df_pred = pd.read_csv(predictions_csv)
    df_pred = df_pred[df_pred["pdb"]==pdb_file.split("/")[-1][:-4]]  # match filename
    df_pred = get_preds_as_str_2_decimal_places(df_pred)

    with open(pdb_file, "r") as original, open(new_pdb_file,'w+') as new:
        for idx, line in enumerate(original):

            # maintain header as-is
            if line[:6] == "REMARK":
                new_line = line
            # check if line of df is present in 
            else:
                chain, imgt, temp = breakdown_line(line)        
                df_match = df_pred[(df_pred["chain_id"]==chain) & (df_pred["IMGT"]==imgt)].reset_index(drop=True)
                # do have paratope predictions for that residue
                if not df_match.empty:
                    pred = df_match["pred_2dp"][0]
                    new_line = line[:21] + chain + line [22:23] + imgt.rjust(3) + line[26:60] + pred.rjust(6) + line[66:]
                # do not have paratope predictions for that residue (set prediction to 0)
                else:
                    new_line = line[:60] + "0.00".rjust(6) + line[66:]
            
            new.write(new_line)

    if inplace:
        os.remove(pdb_file)
        os.rename(new_pdb_file, pdb_file)
