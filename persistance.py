import numpy as np

from config import experiment_scores_path

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

task_names_readable  = [
            "0_tasknames",
            "No Metadata",
            "Task Number/Name",
            "QA1 - Single Supporting Fact",
            "QA2 - Two Supporting Facts",
            "QA3 - Three Supporting Facts",
            "QA4 - Two Arg. Relations",
            "QA5 - Three Arg. Relations",
            "QA6 - Yes/No Questions",
            "QA7 - Counting",
            "QA8 - Lists/Sets",
            "QA9 - Simple Negation",
            "QA10 - Indefinite Knowledge",
            "QA11 - Basic Coreference",
            "QA12 - Conjunction",
            "QA13 - Compound Coreference",
            "QA14 - Time Reasoning",
            "QA15 - Basic Deduction",
            "QA16 - Basic Induction",
            "QA17 - Positional Reasoning",
            "QA18 - Size Reasoning",
            "QA19 - Path Finding",
            "QA20 - Agent's Motivations"]


# FB_LSTM_Baseline = ["10_baseline", "No Metadata", "LSTM Baseline",0.5 ,  0.2 ,  0.2 ,  0.61,  0.7 ,  0.48,  0.49,  0.45,  0.64,
#         0.44,  0.62,  0.74,  0.94,  0.27,  0.21,  0.23,  0.51,  0.52,
#         0.08,  0.91]
# MEMNET_LSTM_Baseline = ["11_baseline", "No Metadata", "SoS Memnet",1.  ,  1.  ,  1.  ,  1.  ,  0.98,  1.  ,  0.85,  0.91,  1.  ,
#         0.98,  1.  ,  1.  ,  1.  ,  0.99,  1.  ,  1.  ,  0.65,  0.95,
#         0.36,  1. ]
# WEAK_MEMNET = ["12_baseline", "No Metadata", "WeS Memnet",0.999,  0.572,  0.236,  0.597,  0.837,  0.49 ,  0.639,  0.622,
#         0.641,  0.313,  0.7  ,  0.899,  0.803,  0.817,  0.352,  0.495,
#         0.491,  0.487,  0.   ,  0.964]
#
# WEAK_MEMNET3 = ["23_baseline", "No Metadata", "PE LS RN JOINT (3-HOP, MemNN)",1.   ,  0.886,  0.781,  0.866,  0.856,  0.972,  0.817,  0.907,
#         0.981,  0.935,  0.997,  0.999,  0.998,  0.931,  1.   ,  0.973,
#         0.596,  0.906,  0.12 ,  1.    ]
#
# WEAK_MEMNET2 = ["22_baseline", "No Metadata", "PE LS RN JOINT (2-HOP, MemNN)",1.   ,  0.844,  0.684,  0.978,  0.866,  0.977,  0.746,  0.883,
#         0.98 ,  0.95 ,  0.988,  1.   ,  0.998,  0.919,  0.995,  0.487,
#         0.588,  0.897,  0.101,  0.999]
#
#
# WEAK_MEMNET_BOW = ["14_baseline", "No Metadata", "BOW 3-Hop",0.994,  0.824,  0.29 ,  0.68 ,  0.817,  0.913,  0.765,  0.886,
#         0.789,  0.772,  0.959,  0.997,  0.895,  0.987,  0.757,  0.48 ,
#         0.546,  0.519,  0.103,  0.999  ]
#
# WEAK_MEMNET1 = ["21_baseline", "No Metadata", " PE LS JOINT (1-HOP, MemNN)", 0.992,  0.38 ,  0.231,  0.772,  0.89 ,  0.928,  0.841,  0.868,
#         0.949,  0.894,  0.916,  0.996,  0.937,  0.631,  0.536,  0.526,
#         0.556,  0.904,  0.093,  1. ]


FB_LSTM_Baseline_10K = ["10_baseline", "No Metadata", "LSTM",24.5 ,  53.2 ,  48.3 ,  0.4,  3.5 ,  11.5,  15.0,  16.5,  10.5,
         22.9,  6.1,  3.8,  0.5,  55.3,  44.7,  52.6,  39.2,  4.8,
         89.5,  1.3]

NTM_JOINT_10K = ["11_baseline", "No Metadata", "NTM",31.5 ,  54.5 ,  43.9 ,  0.0,  0.8 ,  17.1,  17.8,  13.8,  16.4,
         16.6,  15.2,  8.9,  7.4,  24.2,  47.0,  53.6,  25.5,2.2,4.3,1.5]

DNC1_JOINT_10 = ["12_baseline", "No Metadata", "DNC",
0.0,
1.3,
2.4,
0.0,
0.5,
0.0,
0.2,
0.1,
0.0,
0.2,
0.0,
0.1,
0.0,
0.3,
0.0,
52.4,
                 24.1,4.0,0.1,0.0


]


DNC2_JOINT_10 = ["13_baseline", "No Metadata", "DNC2", 0.0,
0.4,
2.4,
0.0,
0.8,
0.0,
0.6,
0.3,
0.2,
0.2,
0.0,
0.0,
0.1,
0.4,
0.0,
55.1,
12.0,
0.8,
3.9,
0.0,]


MEMN2N_JOINT_10 = ["14_baseline", "No Metadata", "MEM2N",
0.0,
1.0,

6.8,
0.0,

6.1,
0.1,

6.6,
2.7,

0.0,
0.5,

0.0,
0.1,

0.0,
                   0.0,
0.2,
0.2,

41.8,
8.0,
75.7,
0.0,]


MEMN2N_JOINT_MEAN_10 = [["15_basemean",0], ["No Metadata",0], ["LSTM - Mean",""],
[28.4, 1.5],
[56.0, 1.5],
[51.3, 1.4],
[0.8, 0.5],
[3.2, 0.5],
[15.2, 1.5],
[16.4, 1.4],
[17.7, 1.2],
[15.4, 1.5],
[28.7, 1.7],
[12.2, 3.5],
[5.4, 0.6],
[7.2, 2.3],
[55.9, 1.2],
[47.0, 1.7],
[53.3 , 1.3],
[34.8, 4.1],
[5.0, 1.4],
[90.9, 1.1],
[1.3, 0.4],
[27.3, 0.8],
[17.1 , 1.0],

]

NTM_JOINT_MEAN_10 = [["16_basemean",0], ["No Metadata",0], ["NTM - Mean",""],
[40.6 , 6.7],
[56.3, 1.5],
[47.8, 1.7],
[0.9, 0.7],
[1.9, 0.8],
[18.4, 1.6],
[19.9, 2.5],
[18.5,4.9],
[17.9, 2.0],
[25.7, 7.3],
[24.4, 7.0],
[21.9, 6.6],
[8.2, 0.8],
[44.9, 13.0],
[46.5, 1.6],
[53.8, 1.4],
[29.9 , 5.2],
[4.5, 1.3],
[86.5, 19.4],
[1.4, 0.6],
[28.5,2.9],
[17.3, 0.7]]



DNC1_JOINT_MEAN_10 = [["18_basemean",0], ["No Metadata",0], ["DNC1 - Mean",""],
[9.0,12.6],
[39.2,20.5],
[39.6,16.4],
[0.4,0.7],
[1.5,1.0],
[6.9,7.5],
[9.8,7.0],
[5.5,5.9],
[7.7,8.3],
[9.6,11.4],
[3.3,5.7],
[5.0,6.3],
[3.1,3.6],
[11.0,7.5],
[27.2,20.1],
[53.6,1.9],
[32.4,8.0],
[4.2,1.8],
[64.6,37.4],
[0.0,0.1],
[16.7,7.6],
[11.2,5.4],


]

DNC2_JOINT_MEAN_10 = [["19_basemean",0], ["No Metadata",0], ["DNC2 - Mean",""],
[16.2, 13.7],
[47.5, 17.3],
[44.3, 14.5],
[0.4 , 0.3],
[1.9, 0.6],
[11.1 , 7.1],
[15.4, 7.1],
[10.0,6.6],
[11.7, 7.4],
[14.7, 10.8],
[7.2, 8.1],
[10.1, 8.1],
[5.5, 3.4],
[15.0, 7.4],
[40.2, 11.1],
[54.7, 1.3],
[30.9, 10.1],
[4.3, 2.1],
[75.8, 30.4],
[0.0 , 0.0],
[20.8 , 7.1],
[14.0, 5.0],


]

def save_scores(values, id):
    #np.array(values)
    if(isinstance(values[0], list)):
        v = values[0][0]
    else:
        v = values[0]
    np.savetxt(experiment_scores_path + v  + "_" + str(id) + ".csv", np.array(values), fmt="%s")

if __name__=="__main__":
    save_scores(task_names_readable,1)
    save_scores(FB_LSTM_Baseline_10K,1)

    #save_scores(NTM_JOINT_10K,1)
    save_scores(DNC1_JOINT_10,1)
    save_scores(DNC2_JOINT_10,10)
    save_scores(MEMN2N_JOINT_10,1)
    #save_scores(MEMN2N_JOINT_MEAN_10,1)
    #save_scores(NTM_JOINT_MEAN_10,1)
    #save_scores(DNC1_JOINT_MEAN_10,1)
    #save_scores(DNC2_JOINT_MEAN_10,1)


    import glob
    from tabulate import tabulate


    len_first = -1
    runs = {}
    for filename in sorted(glob.iglob(experiment_scores_path + "/*.csv")):
        #print filename
        id = filename.split(".")[1].split("_")[-1]
        key = filename.split(id + ".csv")[0]

        column = []
        try:
            column = np.genfromtxt(filename,dtype='str', delimiter="\t")[2:]
        except:
            print filename, "broken"
            pass
        if(len_first == -1):
            len_first = len(column)
        elif((len_first!=len(column) != (len_first+2)!=len(column))):
            print column
            continue

        if(key not in runs):
            runs[key] = [column]
        else:
            runs[key].append(column)

    #print runs.keys()
    columns = []


    for key in sorted(runs.iterkeys()):
        value = runs[key]

        if("tasknames") in key:
            core_data = list(value[0])
            core_data.extend(["Mean Err (%)","Failed(err. > 5%)"])
            columns.append(core_data)
            print len(core_data)

        if("baseline") in key:

            core_data = np.array(value)[:,1:]
            core_data = np.array(core_data, dtype = np.float)
            mn  =  np.mean(core_data, axis = 0)

            lower = np.greater(mn, 5.0)
            lower =  np.sum(lower)

            column = [(value[0][0])] + list(mn)

            column.append("%.1f" %(float(mn.mean()), ))
            column.append("%d" %(lower, ))
            columns.append(column)

        if("basemean") in key:
            core_data = np.array(value)[:,1:]
            new_data = []
            for d in core_data[0]:
                mean, std = d.strip().split(" ")
                new_data.append([float(mean),float(std)])

            new_data = np.array(new_data)
            mn  =  new_data.T[0]
            std =  new_data.T[1]


            #print core_data


            #
            #
            #

            #
            # #exit()
            #
            column = [value[0][0]]
            for m, s in zip(mn, std)[:-2]:
                #line = "%f \pm %f" %(m, s)
                line = "%.1f pm %.1f" %(m, s)

                column.append(line)
            #
            #
            column.append("%.1f pm %.1f" %(mn[-2], std[-2]))
            column.append("%.1f pm %.1f" %(mn[-1], std[-1]))
            columns.append(column)


        elif(len(value) > 1):

            core_data = np.array(value)[:,1:]
            #print core_data
            core_data = 1.0 - np.array(core_data, dtype = np.float)
            core_data = core_data* 100

            std =  np.std(core_data, axis = 0, ddof = 1)
            mn  =  np.mean(core_data, axis = 0)
            mn_overall  =  np.mean(core_data, axis = 1)
            lower = np.greater(core_data, 5)
            #print len(lower), "lowerwe"
            lower =  np.sum(lower, axis = 1)

            #exit()

            column = [value[0][0]]
            for m, s in zip(mn, std):
                #line = "%f \pm %f" %(m, s)
                line = "%.1f pm %.1f" %(m, s)

                column.append(line)


            column.append("%.1f pm %.1f" %(float(mn_overall.mean()), float(mn_overall.std(ddof = 1))))
            column.append("%.1f pm %.1f" %(float(lower.mean()), float(lower.std(ddof = 1))))
            columns.append(column)








    #if(column[0])
    columns = np.array(columns).T
    #print columns.shape



    print tabulate(columns,headers="firstrow", floatfmt=".1f",tablefmt= "latex_booktabs")
    #print tabulate(columns, floatfmt=".1f", headers="firstrow")








