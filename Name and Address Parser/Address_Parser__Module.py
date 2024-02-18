# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 09:15:34 2022

@author: onais
"""

import re
from tqdm import tqdm
import Rulebased as RuleBased
import pandas as pd
import json 
import collections 
import PreprocessingNameAddress as PreProc
import sklearn
from sklearn.metrics import multilabel_confusion_matrix,confusion_matrix,classification_report
#Parsing 1st program
import os
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
today=datetime.today()
current_time = datetime.now().time()

# Format the time as HH:MM:SS
time_string = current_time.strftime("%H:%M:%S")

def Address_Parser(Address_4CAF50,Progress,TruthSet=""):
    Result={}
    RuleBasedOutput={}
    Exception_Mask=""
    FishBone=""
    Detailed_Report=""
    Mask_log={}
    Unique_Mask={}
    Address_4CAF50=open(Address_4CAF50,"r",encoding='utf8')
    file_name = os.path.splitext(os.path.basename(Address_4CAF50.name))[0]
    Lines = Address_4CAF50.readlines()
    fileHandle = open('USAddressWordTable.txt', 'r',encoding='utf8')
    # Strips the newline character
    Observation=0
    Total=0
    Truth_Result={}
    dataFinal={} 
    USAD_Conversion_Dict={
  "USAD_SNO": 1,
  "USAD_SPR": 2,
  "USAD_SNM": 3,
  "USAD_SFX": 4,
  "USAD_SPT": 5,
  "USAD_ANM": 6,
  "USAD_ANO": 7,
  "USAD_CTY": 8,
  "USAD_STA": 9,
  "USAD_ZIP": 10,
  "USAD_ZP4": 11,
  "USAD_BNM": 12,
  "USAD_BNO": 13,
  "USAD_RNM": 14,
  "USAD_RNO": 15,
  "USAD_ORG": 16,
  "USAD_MDG": 17,
  "USAD_MGN": 18,
  "USAD_HNM": 19,
  "USAD_HNO": 20
}
    data={}
    with open('JSONMappingDefault.json', 'r+', encoding='utf8') as f:
        data = json.load(f)
    USAD_CONVERSION_={
        
  "1": "USAD_SNO",
  "2": "USAD_SPR",
  "3": "USAD_SNM",
  "4": "USAD_SFX",
  "5": "USAD_SPT",
  "6": "USAD_ANM",
  "7": "USAD_ANO",
  "8": "USAD_CTY",
  "9": "USAD_STA",
  "10": "USAD_ZIP",
  "11": "USAD_ZP4",
  "12": "USAD_BNM",
  "13": "USAD_BNO",
  "14": "USAD_RNM",
  "15": "USAD_RNO",
  "16": "USAD_ORG",
  "17":"USAD_MDG",
  "18":"USAD_MGN",
  "19":"USAD_HNM",
  "20":"USAD_HNO"
}
    Detailed_Report+="Exception and Mask Report\n"
    ExceptionList = []
    WordTable={}

    for line in fileHandle:
     
        fields=line.split('|')
        WordTable[fields[0]]=fields[1][0]
    Progress.start()
    CNT=100/len(Lines)
    CN=0
    for line in tqdm(Lines):
        CN=CN+CNT
        Progress["value"]=CN
        line=line.strip("\n").split("|")
        ID=line[0].strip()
        try:
            
            line=line[1].strip()
        except:
            continue
        Address=line
        FirstPhaseList=[]
        PackAddress=PreProc.PreProcessingNameAddress().AddresssCleaning(line)
        AddressList=PackAddress[0]
        AddressList = [i for i in AddressList if i]
      
        TrackKey=[]
        Mask=[]
        Combine=""
        LoopCheck=1
        for A in AddressList:
            FirstPhaseDict={}
            NResult=False
            try:
                Compare=A[0].isdigit()
            except:
                print()
            if A==",":
                Mask.append(Combine)
                Combine=""
                FirstPhaseList.append(",")
                #FirstPhaseList.append("Seperator")
            elif Compare:
                Combine+="N"
                TrackKey.append("N")
                FirstPhaseDict["N"]=A
                FirstPhaseList.append(FirstPhaseDict)

            else:
                if A in WordTable.keys():
                    temp=WordTable[A]
                    NResult=True
                    Combine+=temp[0]
                    FirstPhaseDict[temp[0]] = A
                    FirstPhaseList.append(FirstPhaseDict)
                    TrackKey.append(temp[0])
                    
                    
                if NResult==False:
                    Combine+="W"
                    TrackKey.append("W")
                    FirstPhaseDict["W"] = A
                    FirstPhaseList.append(FirstPhaseDict)
            if LoopCheck==len(AddressList):
                Mask.append(Combine)
           
            LoopCheck+=1
            
        Mask_1=",".join(Mask)
        Mask_log[ID]=Mask_1
        Unique_Mask[Mask_1]=ID
        FirstPhaseList = [FirstPhaseList[b] for b in range(len(FirstPhaseList)) if FirstPhaseList[b] != ","]
       
        Found=False
        FoundDict={}

        if Mask_1 in data.keys():
            FoundDict[Mask_1]=data[Mask_1]
            Found=True

        FoundExcept=False
        
        if Found:
            Observation+=1
            Mappings=[]
            for K2,V2 in FoundDict[Mask_1].items():
                FoundDict_KB=FoundDict[Mask_1]
                sorted_Found={k: v for k ,v in sorted(FoundDict_KB.items(), key=lambda item:item[1])}
            for K2,V2 in sorted_Found.items():
                Temp=""
                Merge_token=""
                for p in V2:
                    for K3,V3 in FirstPhaseList[p-1].items():
                       Temp+=" "+V3
                       Temp=Temp.strip()
                       Merge_token+=K3
                       found = False
                       for entry in Mappings:
                           if entry[0] == K2:
                               # Append V3 to existing entry
                               entry[1] += K3
                               entry[2] = ""
                               entry[2] += Temp
                               found = True
                               break
                       if not found:
                         # Add a new entry to Mappings
                           Mappings.append([K2, K3, V3])

            FoundDict_KB=FoundDict[Mask_1]
            sorted_Found={k: v for k ,v in sorted(FoundDict_KB.items(), key=lambda item:item[1])}
            
            
            OutputEntry = {
                "Record ID": ID,
                "INPUT": Address,
                str(Mask_1): Mappings
            }
            OutputList = []
            OutputList.append(OutputEntry)
            # print(OutputList)
            try:
                Truth_Result[ID]=Mappings
                Result[ID]=OutputList
                dataFinal[Mask_1][ID] =Mappings # <--- add `id` value.
                
            except: 
                Result[ID]=OutputList
                Truth_Result[ID]=Mappings
                dataFinal[Mask_1]={}
                dataFinal[Mask_1][ID]=Mappings
                
                
            
        elif not FoundExcept:
            AddressList=[item for item in AddressList if item!=","]
            rules=RuleBased.RuleBasedAddressParser.AddressParser(AddressList)
            ExceptionEntry = {
                "Record ID": ID,
                "INPUT": Address,
                str(Mask_1): rules
            }
            ExceptionList.append(ExceptionEntry)
            RuleBasedOutput[ID]=rules
        else:
            try:
                    
                RuleBasedOutput[ID]=RuleBased.RuleBasedAddressParser.AddressParser(AddressList)
                Exception_Mask+=Mask_1+"\n"
            except:
                continue
        Total+=1
   
    Count_of_Correct=0
    Total_Count=0  
    y_test=[]
    y_predict=[]
    # with open("Individual Address1_Truth_File.txt", 'r+', encoding='utf-8') as g:

    #     Stat = json.load(g)
    #     Count_of_Correct=0
    #     Total_Count=0
                
    #     for key,value in Truth_Result.items():
    #         Total_Count=len(Truth_Result)

    #         if key in Stat.keys():
    #             Count1=0
    #             Count_total=0
    #             for k1,v1 in value.items():
    #                 predict=False
    #                 y_test.append(USAD_Conversion_Dict[k1])
    #                 for k2,v2 in Stat[key].items():
    #                     if v1==v2:
    #                         y_predict.append(USAD_Conversion_Dict[k2])
    #                         Count1+=len(Stat[key])
    #                         predict=True
    #                         break
    
    #                 if not predict:
    #                     for f in v1.split(" "):
    #                         for k2,v2 in Stat[key].items():
    #                             if f in v2:
    #                                 y_predict.append(USAD_Conversion_Dict[k2])
    #                                 predict=True
    #                                 break
    #                         if predict:
    #                             break
    #                         else:
    #                             y_predict.append(0)
    
    
    # Exception_file_name = "_MultiLine_ExceptionFile" + str(current_time) + ".json"
    Exception_file_name = file_name +" "+ str(current_time) + ".json"
    Exception_file_name = re.sub(r'[^\w_. -]', '_', Exception_file_name)
    path = 'Exceptions/MultiLine Exceptions/' + Exception_file_name
    with open(path, 'w', encoding='utf-8') as g:
        g.seek(0)
        json.dump(ExceptionList, g, indent=4)
        g.truncate
    
    FishBone+="Root Cause Analysis"
    if TruthSet!="":
        try:
            with open(TruthSet, 'r+', encoding='utf-8') as g:
                Stat = json.load(g)
                Count_of_Correct=0
                Total_Count=0
                ID=1
                False_Predictions={}
                
                        
                for k in Stat["annotations"]:
                    res=""
                    False_Predictions_Indiv={}
                    
                    for m in k[1].items():
                        
                        for j in m[1]:
                            predict=False
                            y_test.append(USAD_Conversion_Dict[j[2]])
                            Found_Error=False
                            for k1,v1 in Truth_Result[str(ID)].items():
                                if re.sub('\W+','', v1.strip().upper()) == re.sub('\W+','', k[0][j[0]:j[1]].upper().strip()):
                                    
                                    if k1!=j[2]:
                                        Found_Error=True
                                        False_Predictions_Indiv_1={}
                                        False_Predictions_Indiv_1["Correct Class"]=j[2]
                                        False_Predictions_Indiv_1["Incorrect Class"]=k1
                                        False_Predictions_Indiv_1["Value"]=v1
                                        False_Predictions_Indiv["Mask"]=Mask_log[str(ID)]
                                        False_Predictions_Indiv["Raw Address"]=k[0]
                                        False_Predictions_Indiv[str(k1)+"_"+str(j[2])]=False_Predictions_Indiv_1
                                    y_predict.append(USAD_Conversion_Dict[k1])
                                    
                                    predict=True
                                    break
                            if Found_Error:    
                                False_Predictions[ID]=False_Predictions_Indiv
                            
                            if not predict:
                                #y_predict.append(0)
                                y_test.pop()         
                    ID+=1
        except:
            return (False,"Error in the selected file! try again")
        import numpy as np
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        Confusion =  multilabel_confusion_matrix(y_test, y_predict)
        df=classification_report(y_test,y_predict,output_dict=True)
        df_report = pd.DataFrame(df).transpose()
        df_report.reset_index(inplace=True)
        df_report=df_report.replace({"index": USAD_CONVERSION_})
        df_report.to_csv("Metrics.csv")
        
        RTruth=0
        try:
            RTruth=(Count_of_Correct/Total_Count*100)
        except:
            print()
        Detailed_Report+="Output From Active Learning\n\n"
        ActiveLResult = json.dumps(Result, indent = 4,ensure_ascii=False) 
        Detailed_Report+=str(ActiveLResult)
        
        RootCauseReport= json.dumps(False_Predictions, indent=4, ensure_ascii=False)
        
        FishBone+="\n\n"+str(RootCauseReport)
        RuleBasedRes =json.dumps(RuleBasedOutput,indent=4)
        Detailed_Report+="\n\nOutput Fron Rule Based Approach\n\n"
        Detailed_Report+=str(RuleBasedRes)
        Detailed_Report+="\n\nNumber of Exceptions Thrown: -\t"+str(Total-Observation)+"\n"
        Detailed_Report+="Number of Parsed Address: -\t"+str(Observation)+"\n"
        Detailed_Report+="Percentage of Parsed Result: -\t"+str((Observation/Total)*100)+"\n"
        Detailed_Report+="List of Exception Mask(s): -\t\n\n"+Exception_Mask+"--"
        Detailed_Report+="\n\n Evaluation Metrics\n\n"
    
        Detailed_Report+=str(df_report)
        f=open(f"Detailed_Report {file_name}.txt","w",encoding="utf8")
        f1=open(f"Root Cause Report {file_name}.txt","w",encoding="utf8")
        f1.write(FishBone)
        f1.close()
        f.write(Detailed_Report)
        f.close()
        return (True,f"Detailed_Report of {file_name} and Root Cause Report of {file_name} is Generated!")
    else:
        percentage = (Observation/Total)*100
        percentage = "%.2f"% percentage
        ActiveLResult = json.dumps(Result, indent = 4,ensure_ascii=False) 
        # Detailed_Report+="\nNumber of Exceptions Thrown: -\t\t"+"{:,}".format(Total-Observation)+"\n"
        Detailed_Report+="\nNumber of Pattern Parsed Addresses: -\t"+"{:,}".format(Observation)+"\n"
        Detailed_Report+="Percentage of Patterns Parsed Result:  -\t"+"{:.2f}%".format(float(percentage))+"\n"
        Detailed_Report+="\nNumber of Exceptions Thrown: -\t\t"+"{:,}".format(Total-Observation)+"\n"
        Detailed_Report+="Percentage of RuleBased Parsed Result: -\t"+"{:.2f}%".format(100-float(percentage))+"\n"
        Detailed_Report+="Output From Active Learning\n\n"
        Detailed_Report+=str(ActiveLResult)
        
        RuleBasedRes =json.dumps(RuleBasedOutput,indent=4)
        Detailed_Report+="\n\nOutput Fron Rule Based Approach\n\n"
        Detailed_Report+=str(RuleBasedRes)
        # Detailed_Report+="List of Exception Mask(s): -\t\n\n"+Exception_Mask+"--"
        Detailed_Report_1="\nTotal Number of Addresses: -\t"+"{:,}".format(Total)+"\n"
        Detailed_Report_1+="\nUnique Pattern Count: -\t"+"{:,}".format(len(Unique_Mask))+"\n"
        Detailed_Report_1+="\nNumber of Pattern Parsed Addresses: -\t"+"{:,}".format(Observation)+"\n"
        Detailed_Report_1+="Percentage of Patterns Parsed Result:  -\t"+"{:.2f}%".format(float(percentage))+"\n"
        Detailed_Report_1+="\nNumber of Exceptions Thrown: -\t\t"+"{:,}".format(Total-Observation)+"\n"
        Detailed_Report_1+="Percentage of RuleBased Parsed Result: -\t"+"{:.2f}%".format(100-float(percentage))+"\n"
        # Detailed_Report_1+="List of Exception Mask(s): -\t\n\n"+Exception_Mask+"--"
        # Output_file_name = "Detailed_Report_" + str(current_time) + ".txt"
        Output_file_name = "Detailed Report_"+file_name+".txt"
        Output_file_name = re.sub(r'[^\w_. -]', '_', Output_file_name)
        path = 'Output/Batch File Output/' + Output_file_name
        f=open(path,"w",encoding="utf8")
        f.write(Detailed_Report)
        f.close()
        Progress.stop()
        return (True,f"Detailed_Report of {file_name} Generated \n{Detailed_Report_1}")

    # print("Final Correct Address Parsing Percentage",Count_of_Correct/Total_Count*100)
    # print("Address Matching Report")
    # print("Total=",Count)
    # print("Matched Addresses=",Observation)
    # print("Percentage of Matched",(Observation/Count)*100)
            # print("Mask Generated is ",Mask_1)
        # print("Index\tMaskToken\t\tAddress Token")
        # i=1
        # for k in FirstPhaseList:
        #     for key,value in k.items():
        #         print(i,"\t\t",key,"\t\t\t\t",value) 
        #     i+=1
#Address_Parser("D://10k records.txt")
