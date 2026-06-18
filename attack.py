import matplotlib.pyplot as plt
import argparse
import networkx as nx
import time
from dataset import cali_all
from dataset import cali_self
from dataset import nh
import range_attack
import process_database
import numpy
import tqdm
import pickle

def get_correct_edges(p, dictionarry):

    opposite_map = {}

    for i in dictionarry:

        if dictionarry[i] in opposite_map:
            opposite_map[dictionarry[i]].append(i)
        else:
            opposite_map[dictionarry[i]] = [i]

    edges = set()

    for i in tqdm.tqdm(dictionarry):
        nodes_to_add_edges_to = []
        neighbors = []

        if len(dictionarry[i])==3:
            x,y,z = (dictionarry[i][0],dictionarry[i][1],dictionarry[i][2])
            neighbors.append((x+1,y,z))
            neighbors.append((x-1,y,z))
            neighbors.append((x,y+1,z))
            neighbors.append((x,y-1,z))
            neighbors.append((x,y,z+1))
            neighbors.append((x,y,z-1))
        else:
            x,y= (dictionarry[i][0],dictionarry[i][1])
            neighbors.append((x+1,y))
            neighbors.append((x-1,y))
            neighbors.append((x,y+1))
            neighbors.append((x,y-1))        

        for n in neighbors:
            if n in opposite_map:
                nodes_to_add_edges_to.extend(opposite_map[n])

        for n in nodes_to_add_edges_to:
            edges.add((dictionarry[i],dictionarry[n]))
            edges.add((dictionarry[n],dictionarry[i]))


    return edges
def check_accuracy_with_edges(G,dictionarry, points):
    correct = 0
    incorrect = 0
    edges = G.edges()
    correct_edges = set()

    max_correct =get_correct_edges(points, dictionarry) 

    for i in edges:
        correct_edge = False

        front_nodes = i[0]
        end_nodes = i[1]

        dist = 0
        s = 0

        good = True

        point = dictionarry[front_nodes[0]] 
        for f in list(front_nodes):
            if point!=dictionarry[f]:
                good = False
        point = dictionarry[end_nodes[0]] 
        for e in list(end_nodes):
            if point!=dictionarry[e]:
                good = False   

        if good:
            translated_node = (dictionarry[f] , dictionarry[e] ) 
            if numpy.linalg.norm(numpy.array(translated_node[0])-numpy.array(translated_node[1]))  <= 1:
                correct_edge = True

        if correct_edge:
            if (translated_node not in correct_edges) and ((translated_node[1], translated_node[0]) not in correct_edges):
                correct_edges.add(translated_node)
                correct_edges.add((translated_node[1],translated_node[0]))
        else:
            incorrect+=1

    precision = len(correct_edges)/(len(correct_edges)+incorrect)
    recall = len(correct_edges)/len(max_correct)

    return precision,recall


def plot3D(pos):
    X = []
    Y = []
    Z = []

    for i in pos:
        point = pos[i]
        X.append(point[0])
        Y.append(point[1])
        Z.append(point[2])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(X, Y, Z, s=5)
    plt.gca().set_aspect('equal')
    plt.show()

def plot2D(pos):
    X = []
    Y = []

    for i in pos:
        point = pos[i]
        X.append(point[0])
        Y.append(point[1])

    fig = plt.figure()

    plt.scatter(X, Y, s=5)
    plt.gca().set_aspect('equal')
    plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Dense Attacks!')
    parser.add_argument('-points', type=str, default = "small_grid",
                        help='cali_50, grid, dg, crg, nh, boat')

    parser.add_argument('-p', type=float, default = 100,
                        help='percentage of queries')

    parser.add_argument('-graph_drawing', type=str, default = "dense",
                        help='kk, tms')   
    parser.add_argument('-dist', type=str, default = "uniform",
                        help='beta, gaussian, uniform') 
    parser.add_argument('-N0', type=int, default = 20,
                        help='N0')
    parser.add_argument('-N1', type=int, default = 20,
                        help='N1') 

    args = parser.parse_args()


    if args.points == "cali_50":
        cali_all = process_database.scale_points(cali_all,50,50)
        points, map_to_original, N0,N1 = process_database.make_database_from_points(cali_all)
    elif args.points == "cali_self":
        cali_self = process_database.scale_points(cali_self,50,50)
        points, map_to_original, N0,N1 = process_database.make_database_from_points(cali_self)                
    elif args.points == "grid":
        N0 = int(args.N0)
        N1 = int(args.N1)
        points, map_to_original = process_database.get_random_database(N0,N1, 1)
    elif args.points == "nh":
        points, map_to_original, N0,N1,N2 = process_database.make_database_from_points_3D(nh)
    elif args.points == "dg":
        with open("datasets/dg", 'rb') as f:
            schools_small = pickle.load(f)
        points, map_to_original, N0,N1 = process_database.make_database_from_points(schools_small)
    elif args.points == "crg":
        schools_3d_smaller_45 = []
        with open("datasets/crg", 'rb') as f:
            schools_3d_smaller_45 = pickle.load(f)
        points, map_to_original, N0,N1,N2 = process_database.make_database_from_points_3D(schools_3d_smaller_45)
    elif args.points == "boat":
        with open("datasets/boat.pickle", 'rb') as f:
            boat = pickle.load(f)
        points, map_to_original, N0,N1 = process_database.make_database_from_points(boat)
    else:
        print("I don't know this dataset")
        exit()

    print("Getting Responses")
    if "nh" in args.points or "crg" in args.points:
        responses = process_database.get_responses_no_vals_3D(points,map_to_original, N0, N1,N2)

    else:
        responses = process_database.get_responses_no_vals(points,map_to_original, N0, N1)

    print("Sampling Responses")
    unique_rs = set()


    if args.dist == "uniform":
        new_responses = process_database.sample_uniform(responses, int(len(responses)*args.p/100.0))
    elif args.dist == "beta":
        new_responses = process_database.sample_beta(responses, int(len(responses)*args.p/100.0))
    elif args.dist == "gaussian":
        new_responses = process_database.sample_gaussian(responses, int(len(responses)*args.p/100.0))
    else:
        print("I don't know that distribution")
        exit()

    if not("nh" in args.points or "crg" in args.points or "3d" in args.points):
        new_responses, unique_rs = process_database.get_actual_resps_after_sampling(new_responses,points,map_to_original)
    else:

        new_responses, unique_rs = process_database.get_actual_resps_after_sampling_3D(new_responses,points,map_to_original)

    start = time.time()
    G, used = range_attack.general(new_responses)
    end = time.time()
    
    precision,recall = check_accuracy_with_edges(G,map_to_original,points)
    print("Attack Precision: ", precision, " , Recall: ", recall, "Time: ", str(end-start))



    if "nh" in args.points or "crg" in args.points:
        dim = 3 
        pos = nx.kamada_kawai_layout(G, dim=dim)
        plot3D(pos)
    else:
        dim = 2
        pos = nx.kamada_kawai_layout(G, dim=dim)
        plot2D(pos)



        












