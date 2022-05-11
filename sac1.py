from igraph import *
import numpy as numpy
import pandas as pandas
import math
import sys

# Considering cosine similarity between two vectors (v1,v2) with different attributes
# Cosine similarity = (dot product)/(|v1|*|v2|)
def cosineSimilarity(vector1,vector2):

    modVector1 = 0
    modVector2 = 0
    dotProduct = 0
    
    vector1List = list(vector1.attributes().values())
    vector2List = list(vector2.attributes().values())

    vectorLength = len(vector1List)

    for i in range(vectorLength):
        modVector1 = modVector1 + math.pow(vector1List[i],2)
        modVector2 = modVector2 + math.pow(vector2List[i],2)
        dotProduct = dotProduct + vector1List[i]*vector2List[i]
    
    return dotProduct/math.sqrt(modVector1*modVector2)

# Compute similarity matrix between each pair of vertices
# Calling cosine similarity metric for each pair of vertices
def computeSimilarityMatrix(graph):

    vertices = len(graph.vs)
    similarityMatrix = [[None for i in range(vertices)] for j in range(vertices)]

    for i in range(vertices):
        for j in range(vertices):
            if similarityMatrix[i][j] is not None:
                continue
            else:
                similarityMatrix[i][j] = cosineSimilarity(graph.vs[i],graph.vs[j])
    return similarityMatrix

# This method calculates DetlaQ_Newman which is the structural modulairty
def delta_Q_newman(graph,communityList,alpha,vertex,community_j):
    
    # Old modularity
    prev_modularity = graph.modularity(communityList, weights='weight')
    tempC = communityList[vertex]
    communityList[vertex] = community_j
    # New modularity
    new_modularity = graph.modularity(communityList, weights='weight')
    communityList[vertex] = tempC
    # Delta_Q_newman change
    return new_modularity - prev_modularity

# This method calculayes DeltaQAttr which is the sum of similarity Sim(i,x) for all i belongs to a community
def delta_Q_attr(graph,vertex,community_j,similarityMatrix,communityList):
    sum=0
    count = 0
    for i in range(len(communityList)):
        if communityList[i] == community_j:
            sum += similarityMatrix[vertex][i]
            count+=1
    # Normalizing it
    return sum /(count*len(set(communityList)))
# This method implements phase 1 of the SAC1 algorithm
def phase1(graph,alpha,similarityMatrix):
    
    communityList = [i for i in range(len(graph.vs))]
    # Fixed number of iterations
    iterations = 15
    currentIteration = 0
    vertices = graph.vs
    loop_break = False
    while not loop_break and currentIteration < iterations:
        print('iteration',currentIteration)
        for i in range(len(vertices)):
            vertex = i
            community_i = communityList[vertex]
            max_delta_Q = -math.inf
            max_community = None
            print('vertex:',vertex)
            for community_j in communityList:
                if community_i == community_j:
                    continue
                # Calculate Delta_Q: which is (alpha*Delta_Q_Newman + (1-alpha)*Delta_Q_Attr)
                delta_Q = alpha*delta_Q_newman(graph,communityList,alpha,vertex,community_j)+(1-alpha)*delta_Q_attr(graph,vertex,community_j,similarityMatrix,communityList)
                
                # If delta_Q is greater than max_delta_Q
                # Change the max_community
                if max_delta_Q < delta_Q:
                    max_delta_Q = delta_Q
                    max_community = community_j
            
            # Assign max_community to the vertex 
            if max_delta_Q > 0 and max_community:
                communityList[vertex] = max_community
                loop_break = False
            # If the max_delta_Q is less than or equal to 0 then break loop
            elif max_delta_Q <= 0:
                loop_break = True
        # Increment iteration
        currentIteration = currentIteration+1
    return communityList

# This method implements phase 2 of the SAC1 algorithm
def phase2(graph,communityList,alpha,similarityMatrix):
    # The clusters have to be rebased so that they start from index 0
    communityList, mapped_clusters = rebaseClusters(communityList)

    # Using contract_vertices and simplify to combine vertices
    # Using mean metric to calculate the common attributes for the cluster
    graph.contract_vertices(communityList,combine_attrs="mean")
    graph.simplify(combine_edges=sum,multiple=True,loops=True)

    # Calculate similarity matrix again for the new set of vertices
    similarityMatrix = computeSimilarityMatrix(graph)

    # Adding node to the communities
    communityList = [i for i in range(len(graph.vs))]

    # Running the phase 1 again
    communityList = phase1(graph,alpha,similarityMatrix)
    return communityList, mapped_clusters

# Rebasing the clusters from index 0
def rebaseClusters(list):
    
    newMapping = {}
    # new community list
    newCommunityList = []
    count = 0
    
    mappedClusters = {}
    for i in range(len(list)):
        vertex = list[i]
        if vertex in newMapping:
            newCommunityList.append(newMapping[vertex])
            mappedClusters[newMapping[vertex]].append(i)
        else:
            newMapping[vertex] = count
            newCommunityList.append(count)
            mappedClusters[count] = [i]
            count+=1
    return newCommunityList, mappedClusters

# Main function
def main(alpha):

    # Converting into floating point number
    alpha = float(alpha)
    # Reading edge list from the data folder
    graph = Graph.Read_Edgelist('data/fb_caltech_small_edgelist.txt')
    # Reading csv file from the data folder
    attrList = pandas.read_csv('data/fb_caltech_small_attrlist.csv')
    # Each edge weighs 1
    graph.es['weight'] = 1
    # Field names of all the attributes
    attrFields = attrList.keys()
    
    # Assigning attribute fields and values to the vertices of graphs 
    for i in attrList.iterrows():
        for j in range(len(attrFields)):
            
            graph.vs[i[0]][attrFields[j]] = i[1][j]

    # Computing Similarity Matrix
    similarityMatrix = computeSimilarityMatrix(graph)
    # Phase1 result 
    phase1_community_list = phase1(graph,alpha,similarityMatrix)
    
    # Phase2 result
    phase2_community_list, mappedClusters = phase2(graph,phase1_community_list,alpha,similarityMatrix)
    
    phase2_communities_output = ''
    #Getting the result in the desired format using VertexClustering
    for cluster in VertexClustering(graph,phase2_community_list):
        if cluster:
            originalVertices = []
            for vertex in cluster:
                originalVertices.extend(mappedClusters[vertex])
            phase2_communities_output += ','.join(map(str,originalVertices))
            phase2_communities_output += '\n'
            print(cluster)
    phase2_communities_output = phase2_communities_output[:-2]

    # Using alpha File name
    alphaFileName = 0
    if alpha == 0.0:
        alphaFileName = 0
    elif alpha == 1.0:
        alphaFileName = 1
    elif alpha == 0.5:
        alphaFileName = 5
    # File opening
    file = open("communities_"+str(alphaFileName)+".txt", 'w+')
    # File writing
    file.write(phase2_communities_output)
    #File closing
    file.close()
    return



if __name__=="__main__":
    # 2 arguments are required for this file to execute
    if len(sys.argv) == 2:
        alpha = sys.argv[1]
    else:
        print("Please run again by including alpha value")
        exit()

    main(alpha)