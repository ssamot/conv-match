import seaborn as sns
sns.set(style="darkgrid")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


experiment = 0
data = []
with open("out_all.txt") as f:
    epoch = 0

    for i, line in enumerate(f):
        if(line.startswith("Epoch")):
            #print i,line
            epoch = int(line.split("/")[0][6:])
            # if(epoch==50):
            #     print epoch, experiment
            #     experiment+=1


        splitted_line = line.split("val_loss")
        #print splitted_line
        if(len(splitted_line) > 1):
            #print len(splitted_line)
            #print splitted_line[0]
            #print ("---------------")
            space_splitted = splitted_line[0].split(" ")
            index = len(space_splitted) - 1 - space_splitted[::-1].index("loss:")
            #print splitted_line
            data.append([epoch, float(space_splitted[index+1]), experiment, "128"])
            #data.append([epoch, float(space_splitted[index+1])-1, experiment, "64"])
            if(epoch==50):
                experiment+=1
            # if(experiment) > 3:
            #     break


column_names = ["iteration", "cross-entropy loss", "try", "algorithm"]

df = pd.DataFrame(data, columns = column_names)

print df

#print df
#exit()

# Plot the response with standard error
ax = sns.tsplot(data=df, time="iteration", unit="try", value="cross-entropy loss", ci = 99)

#print gammas
markers=[',', '>', '<', "D", '.', 'o', '*', "s", "v"]

#ax.set(ylim=(0, None))


for i in range(len(ax.lines)):
    ax.lines[i].set_marker(markers[i])

ax.legend()

plt.savefig("plot" + ".pdf")








