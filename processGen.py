# -*- coding: utf-8 -*-
"""
Created on Thu May 27 00:01:48 2021

@author: Lyle
"""

import dataFrameTools
import convertParams
import pandas as pd
from pathlib import Path
import checkdf

def processGen(file="synthesized", gen_name="", dataset="", genbcad=1, denorm = False, check=True, from_OH=True, intermediates=0, sourcepath = "PlainRoadbikestandardized.txt", targetpath = "../Generated BCAD Files/Files/"): 
    if isinstance(file, str):
        df=pd.read_csv(Path("../data/"+file+".csv"), index_col=0)
    else:
        df=file
        
    print(f"\n--- DEBUG PIPELINE START ---")
    print(f"1. Initial DF shape: {df.shape}, Index: {df.index.tolist()}")

    if denorm:
        df=dataFrameTools.deNormalizeDF(df, dataset, 1, intermediates)
        
    if check: #-1 for use all
        df = checkdf.checkdf(df, gen_name, 1, intermediates)
        print(f"2. After checkdf shape: {df.shape}")
        
    if from_OH:
        df=dataFrameTools.deOH(df, dataset, intermediates)
        print(f"3. After deOH shape: {df.shape}")
        
    reDF=dataFrameTools.convertOneHot(df, dataset, 0)
    reDF=dataFrameTools.standardizeReOH(reDF, dataset, intermediates)    
    
    if genbcad==1:
        deOHdf=convertParams.deconvert(df, dataset)
        print(f"4. After convertParams shape: {deOHdf.shape}")
        # Let's peek at a core value to see if it turned into NaN!
        if 'Top tube rear diameter' in deOHdf.columns:
            print(f"   Value of 'Top tube rear diameter': {deOHdf.at[deOHdf.index[0], 'Top tube rear diameter']}")
        
        print("5. Sending to genBCAD...")
        dataFrameTools.genBCAD(deOHdf, sourcepath, targetpath)
        print(f"--- DEBUG PIPELINE END ---\n")
        
    return reDF

    