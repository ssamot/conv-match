import numpy as np

experiment_scores_path  = "./experiments_arch"

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



def save_scores(values, id):
    np.savetxt(experiment_scores_path + values[0]  + "_" + str(id) + ".csv", np.array(values), fmt="%s")

if __name__=="__main__":



    import glob
    from tabulate import tabulate

    columns = []
    len_first = -1
    for filename in sorted(glob.iglob(experiment_scores_path + "/*.csv")):
        #print filename
        column = []
        try:
            column = np.genfromtxt(filename,dtype='str', delimiter="\t")[2:]
        except:
            print filename, "broken"
            pass
        if(len_first == -1):
            len_first = len(column)
        elif(len_first!=len(column)):
            continue

        columns.append(column)

    columns = np.array(columns).T
    c_calc =  100 - 100*np.array(columns.T[1:,1:], dtype = np.float)
    columns.T[1:,1:] = c_calc
    mean = ["Mean Err (%)"] + list(c_calc.mean(axis = 1))
    higher = np.greater(c_calc, 5)

    higher =  np.sum(higher, axis = 1)
    print higher
    greater = ["Failed(err. > 5%)"] + list(higher)
    print greater
    print mean

    columns = np.vstack([columns,mean,greater])
    #if(column[0])

    print tabulate(columns,headers="firstrow", tablefmt= "latex_booktabs", floatfmt=".1f")
    #print tabulate(columns,headers="firstrow", tablefmt= "html")





