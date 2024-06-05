import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets
from random import randint,choices,shuffle
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import silhouette_score
from os import system

class genetic:
    def __init__(self,count:int,data:np.ndarray,cluster_size:int=(2,4),dynamic_clustering=True) -> None:
        """if dynamic clustering is true,cluster size must be a tuple,containing cluster size range
        else it must be an int number"""
        self.data=data
        self.data_count=data.shape[0]
        self.genome_count=count
        self.genomes=[]
        
        for i in range(count):
            if dynamic_clustering:
                chosen_cluster_size=randint(cluster_size[0],cluster_size[1])
            else:
                chosen_cluster_size=cluster_size

            self.genomes.append([])
            
            for _ in range(self.data_count):
                self.genomes[i].append(randint(1,chosen_cluster_size))
            self.genomes[i].append(chosen_cluster_size)
            
            

    def fitness(self):
        result=[]
        
        for i in range(self.genome_count):
            #print(self.genomes[i])
            result.append(silhouette_score(self.data,self.genomes[i][:-1]))
            
        result=minmax_scale(result,(1,10))
        return result
    
    
    
    def crossover(self,child_population_range:tuple):
        next_gen=[]
        
        fitness_result=self.fitness()
        child_population=randint(child_population_range[0],child_population_range[1])
        parent1=choices(list(range(self.genome_count)),weights=list(fitness_result),k=child_population)
        parent2=choices(list(range(self.genome_count)),weights=list(fitness_result),k=child_population)
        temp=self.data_count//2
        
        for x,y in tuple(zip(parent1,parent2)):
            child=[]
            pick_order=[0 for _ in range(temp)]+[1 for _ in range(self.data_count-temp)]
            shuffle(pick_order)
            
            for i in range(self.data_count):
                if pick_order[i]:
                    child.append(self.genomes[x][i])
                else:
                    child.append(self.genomes[y][i])
            child.append(max(self.genomes[y][-1],self.genomes[x][-1]))
            
            index_1=randint(0,self.data_count-1)
            index_2=randint(0,self.data_count-1)
            child[index_1],child[index_2]=child[index_2],child[index_1]
            #child[randint(0,self.data_count-1)]=randint(1,child[-1])
            
            next_gen.append(child)
            
        self.genomes=next_gen
        self.genome_count=child_population
        
    def clustering(self,generations:int,child_population_range:tuple):
        
        for i in range(generations):
            system('cls')
            percent=(i*100)//generations
            print(f"clustering_progress:{percent}%\n{percent*'='}")
            self.crossover(child_population_range)
        
        final_fitness=self.fitness()
        highest_index=0
        
        for i in range(1,self.genome_count):
            if final_fitness[i]>final_fitness[highest_index]:
                highest_index=i
        print("clustering Done")
        
        return self.genomes[highest_index][:-1]
        
            
        
        
iris_data,currect_clustering= datasets.load_iris(return_X_y=True)

population=genetic(200,iris_data,dynamic_clustering=False,cluster_size=3)
result=population.clustering(200,(150,300))

    



fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

X_reduced = PCA(n_components=3).fit_transform(iris_data)
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=np.array(result),
    s=40,
)

ax.set_title("First three PCA dimensions")
ax.set_xlabel("1st Eigenvector")
ax.xaxis.set_ticklabels([])
ax.set_ylabel("2nd Eigenvector")
ax.yaxis.set_ticklabels([])
ax.set_zlabel("3rd Eigenvector")
ax.zaxis.set_ticklabels([])
plt.show()






























# def old_fitness(self):
#         result={}
#         for j in range(self.genome_count):
#             clusters_data={}
#             result[j]=0
#             for i in range(1,self.genomes[j][-1]+1):
#                 clusters_data[i]=[]
                
#             for i in range(self.data_count):
#                 clusters_data[self.genomes[j][i]].append(self.data[i])
            
#             for i in range(1,self.genomes[j][-1]+1):
#                 if len(clusters_data[i]) <2:
#                     continue
#                 else:
#                     temp=np.std(np.array(clusters_data[i]))
#                     result[j]+=10*(1/(temp+1))
#         return result