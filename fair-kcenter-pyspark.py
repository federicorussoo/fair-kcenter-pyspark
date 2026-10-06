import random
import math
from pyspark import SparkContext, SparkConf
import time
import sys
import array

euclidean_dist = math.dist


def FairFFT(U, k_A, k_B):
    '''
    Fair Farthest First Traversal (Fair-FFT) algorithm
    Selects k_A points of label 'A' and k_B points of label 'B' by maximizing 
    the minimum distance from the already selected set S

    Input:
    U:      list of tuples where the first element is a tuple containing point's coordinates and the second element is a char representing point's labels
            [ ((x_1, x_2, ..., x_D), 'A'), ... ,((y_1, y_2, ..., y_D), 'B') ]
    k_A:    number of centroids to be found with label A
    k_B:    number of centroids to be found with label B

    Output:
    S:      a list of selected centroids with the same format of U
    '''

    l=len(U)
    S=[]

    # Random choice of the first centroid 
    while True:
        index_c = random.randint(0, l-1)
        coord_c, label_c = U[index_c]
        if k_A == 0 and label_c == 'A':
            continue
        if k_B == 0 and label_c == 'B':
            continue
        break

    S.append(U[index_c])

    # Initialize an array to store the minimum distance of each point from the set S
    min_dist = array.array('d', [float('inf')] * l)
    # Set the distance of the chosen centroid to -inf so it won't be selected again
    min_dist[index_c] = -float('inf')

    if label_c=='A':
        count_A=1
    else:
        count_A=0
    if label_c=='B':
        count_B=1
    else:
        count_B=0

    while count_A<k_A or count_B<k_B:
        
        need_A = count_A < k_A
        need_B = count_B < k_B

        max_dist=-float('inf')
        max_index=-1

        for index in range(l):

            # given that points in S (centroids) have min_dist set to -inf they are skipped
            if min_dist[index] < 0.0:
                continue


            # even though euclidean_dist (which is math.dist) computes the square root of the sum of squared differences of coordinates
            # it runs significantly faster than a custom made function computing squared distance of the points
            distance = euclidean_dist(coord_c, U[index][0])

            if distance < min_dist[index]:
                min_dist[index] = distance

            # Update max candidate only if its class budget (k_A or k_B) is not yet exhausted
            if min_dist[index]>max_dist:
                label_p = U[index][1]
                if (label_p == 'A' and need_A) or (label_p == 'B' and need_B):
                    max_dist = min_dist[index]
                    max_index = index


        coord_c, label_c = U[max_index]
        S.append(U[max_index])
        min_dist[max_index]=-float('inf')

        if label_c=='A':
            count_A+=1
        else:
            count_B+=1

    return S

def parsing_line(line):
    '''
    Parses a CSV-formatted string into a point with coordinates and a label

    Input:
    line:   A comma-separated string (e.g., "1.5,3.3,B")

    Output: 
    tuple: ((coord_1, coord_2, ...), label)
    '''
    parts = line.split(",")
    coords = tuple(float(x) for x in parts[:-1])
    label = parts[-1]
    return (coords, label)

def MRFairFFT(inputPoints, k_A, k_B, mult, NA, NB):
    '''
    2-Rounds MapReduce implementation of the Fair-FFT algorithm:
        - Round 1 (Map) extracts local coresets from partitions
        - Round 2 (Reduce) produces the final global coreset

    Input:
    inputPoints: RDD containing tuples ((coords), label)
    k_A, k_B:    number of centers required for each label
    mult:        oversampling factor for the first round
    NA, NB:      number of points with label A and label B in the dataset
    '''
    k_A_round1 = round(mult * k_A)
    k_B_round1 = round(mult * k_B)

    if NA == 0 and NB == 0:
        raise ValueError("The input dataset U is completely empty (NA=0, NB=0).") 

    if k_A == 0 and k_B == 0:
        raise ValueError("Invalid parameters: k_A and k_B cannot be both equal to 0.")
        
    if k_A == 0 and NB == 0:
        raise ValueError("Invalid state: k_A is set to 0, but U_b is empty. Impossible to pick k_B centers.")

    if k_B == 0 and NA == 0:
        raise ValueError("Invalid state: k_B is set to 0, but U_a is empty. Impossible to pick k_A centers.")
    
    # ROUND 1: MAP
    # ROUND 1: REDUCE (empty)
    # ROUND 2: MAP (empty)
    # ROUND 2: REDUCE
    partial_coresets = inputPoints.mapPartitions(lambda partition: FairFFT(list(partition), k_A_round1 , k_B_round1))                                                                                                                                                                                                         
                                                                                                                        
    final_coreset = FairFFT(partial_coresets.collect(), k_A , k_B)                                                      
    return final_coreset

def compute_objective_fun(point, S):
    '''
    Computes the distance of a point from the set S

    Input:
    point:  A tuple ((coords), label)
    S:      list of selected centroids [((coords), label), ...]

    Output:
    min_dist: the Euclidean distance to the closest centroid in S
    '''
    
    min_dist=float('inf')

    for center in S:

        dist_temp = euclidean_dist(center[0],point[0])
        
        if  dist_temp < min_dist:
            min_dist = dist_temp
     
    return min_dist

    
def update_partition_counts(current_acc, point):
    '''
    Updates the partition's local counter based on the point's label
    Used by aggregate to process points sequentially within each partition

    Input:
    current_acc: Tuple (count_A, count_B) accumulated so far in the partition
    point:       The current tuple ((coords), label) being processed

    Output:
    A tuple (updated_count_A, updated_count_B)
    '''
    label = point[1]
    res_a = current_acc[0] + (1 if label == 'A' else 0)
    res_b = current_acc[1] + (1 if label == 'B' else 0)
    return (res_a, res_b)


def merge_partition_counts(local_acc1, local_acc2):
    '''
    Sums the local counters to produce the final global counts (NA, NB)
    Used by aggregate to merge results from different partitions
    
    Input:
    local_acc1:  Tuple (count_A, count_B) from one partition
    local_acc2:  Tuple (count_A, count_B) from another partition

    Output:
    A tuple (total_A, total_B) representing the combined sum
    '''
    total_a = local_acc1[0] + local_acc2[0]
    total_b = local_acc1[1] + local_acc2[1]
    return (total_a, total_b)


def main():
    '''
    Execution of the following steps:
        1. Partitioning and parsing
        2. Counting of the number of points with label A and B
        3. Two-round MapReduce to extract the final global coreset
        4. Parallel evaluation of the objective function 
    '''

    assert len(sys.argv) == 5, "Usage: python G61HW1.py <file_name> <k_A> <k_B> <L>"

    conf = SparkConf().setAppName('G61HW1')
    sc = SparkContext(conf=conf)

    data_path = sys.argv[1]
    k_A = int(sys.argv[2])
    k_B = int(sys.argv[3])
    L = int(sys.argv[4])

    # We tested oversampling the number of centroids extracted in Round 1 using a multiplier > 1 (e.g., 2, 5, 10, 20, 50, 100).
    # Results on a 1.3M points dataset showed that increasing the multiplier causes execution 
    # time to grow significantly (from 476 ms at mult = 1 to >20s at mult = 100), 
    # without any significant improvement in the objective function (fluctuating around 39).
    # Since Farthest First Traversal naturally targets the borders of the dataset, 
    # extracting more points in Round 1 increases the probability of sampling inner points,
    # slowing down the frontend node in Round 2 without providing more representative centroids.
    # Therefore, we set te multiplier to 1 to maximize computational efficiency.
  
    mult = 1

    print(f"File path = {data_path}, KA = {k_A}, KB = {k_B}, L = {L}")
    
    # STEP 1
    # Load, parse, and partition data into L groups
    # .cache() is used to avoid recomputation
    inputPoints = sc.textFile(data_path).map(parsing_line).repartition(numPartitions = L).cache()

    # STEP 2
    # Computes NA and NB across partitions without pulling the entire dataset to the frontend node
    NA, NB = inputPoints.aggregate((0, 0), update_partition_counts, merge_partition_counts)
    print(f"N = {NA + NB}, NA = {NA}, NB = {NB}")
    
    # STEP 3
    start_time = time.time()
    S = MRFairFFT(inputPoints, k_A, k_B, mult, NA, NB)
    execution_time = time.time() - start_time

    for center, label in S:
        formatted_coords = ",".join(str(coord) for coord in center)
        print(f"Center = [{formatted_coords}] Label = {label}")
    
    # STEP 4
    # Broadcast the small set S to all workers to minimize network overhead in the parallel distance computation
    S_broadcast = sc.broadcast(S)

    objective_function_val = inputPoints.map(lambda point: compute_objective_fun(point, S_broadcast.value)).max()
    print(f"Objective function = {objective_function_val}")

    time_in_ms = int(execution_time * 1000)
    print(f"Running time of MRFairFFT = {time_in_ms} ms")

if __name__ == "__main__":
    main()
