# libs on libs
#import suds
#from suds.xsd.doctor import Import, ImportDoctor
import datetime
from datetime import datetime, timedelta, time
from nltk import bigrams
from nltk import trigrams
import socket
import threading
import base64
import statistics
import time, sys
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA

# custom modules
import vector_creation

#globals
debug_flag=1
master_log_entry_list = []


#define a struct to hold the information we parse out of each NXDomain entry.
class logEntry():
    def __init__(self, src, request_url):
        self.src_ip = src
        self.request_url = request_url        
        self.overlap_vector = []
        self.overlap_label = None
        self.statistical_cluster_label = None
        self.final_cluster_ID = None

        # functions to make our object itterable.
        def __iter__(self):
            return self

        def __next__(self):
            return self.next

        # a hash function to let us compare log entry objects
        def __hash__(self):
            return hash(self.request_url)


# define a struct to be used in the NXk domain subsets
class nxkDomainSubset():
    def __init__(self, log_entries):
        self.log_entries = log_entries
        self.feature_vector = []
        self.fv_label = None


# in this function, we make use of the cluster_id we associated to each of the log entries.
# you'll remember that we assigned two of these, one from the similarity matrix clustering and one 
# from the statistic similarity clustering. We will take the intersect of all possible pairs, creating 
# a final list of corrolated clusters that reach our size threashold.
def clusterCorrelation(log_entries, num_of_lables_stats, 
    num_of_lables_matrix, threashold_floor, threashold_ceiling):
    print("\n\n┻━┻︵  \(°□°)/ ︵ ┻━┻  . ")
    print("!!! BEGIN CLUSTER CORROLATION !!!")

    #place holders for our two lists of cluster sets
    statistical_cluster_list = [] 
    bipartite_graph_list = []
    corrolated_clusters = []

    #append new blank lists to our sets, One for every cluster in the respective cluster type. 
    for x in range(0, num_of_lables_stats):
        statistical_cluster_list.append(set())

    for x in range(0, num_of_lables_matrix):
        bipartite_graph_list.append(set())

    #go through all of the log entries to craft 2 sets of lists based on the clusters they belong to.
    for entry in log_entries:
        statistical_cluster_id = entry.statistical_cluster_label
        similarity_matrix_cluster_id = entry.overlap_label

        if(statistical_cluster_id != None):
            statistical_cluster_list[statistical_cluster_id].add(entry)
        
        if(similarity_matrix_cluster_id != None):    
            bipartite_graph_list[similarity_matrix_cluster_id].add(entry)
    
    #for statistical_set_obj in statistical_cluster_list:
        #for log_entry in statistical_set_obj:
            #print(log_entry.request_url) 

    #now that we have all of our clusters in memory, lets perform some intersect operations on our sets of clusters.
    #we will only keep those that match our size threashold.
    for statistical_set_obj in statistical_cluster_list:
        for matrix_set_obj in bipartite_graph_list:
            intersect_set = statistical_set_obj | matrix_set_obj

            if(len(intersect_set) > threashold_floor and len(intersect_set) < threashold_ceiling):
                corrolated_clusters.append(list(intersect_set))

    print("\n\n┻━┻︵  \(°□°)/ ︵ ┻━┻  ┻━┻︵  \(°□°)/ ︵ ┻━┻ ")
    print("CORROLATION COMPLETED, ALL YOUR BASE ARE BELONG TO US.")

    return corrolated_clusters



# implementation of x-means clustering with the help of scikit.
# Q's: | what should our n_cluster size be? | how larger should batch_size be | what about other tweaks to k-means? |
def xmeansClusteringLogEntries(log_entries):
    print("\n\n┻━┻︵  \(°□°)/ ︵ ┻━┻ ")
    print("START CLUSTERING OUR DENSE VECTORS FROM THE OVERLAP MATRIX")
    # our new numpy array that has a dimension representing the size of the dense vector
    # created from our bipartite matrix. They should all be uniform so any single
    # sample will do.
    numpy_fv_array = np.zeros(shape=(len(log_entries), len(log_entries[0].overlap_vector)))

    # craft a master numpy array of all feature vectors. 
    index = 0
    for entry in log_entries:
        numpy_fv_array[index] = entry.overlap_vector
        index += 1

    # build our kmeans object
    #km = MiniBatchKMeans(n_clusters=10, init='k-means++', max_iter=300, n_init=10, verbose=0, batch_size=10000)
    km = MiniBatchKMeans(init='k-means++', max_iter=300, verbose=0, batch_size=10000)

    # fit our feature vectors to lables.
    t0 = time.time()
    label_set = km.fit_predict(numpy_fv_array)
    num_of_labels = len(np.unique(label_set))
    print("done in %0.3fs" % (time.time() - t0))

    # now apply those lables back into the subset objects for future cluster corrolation.
    index = 0
    for entry in log_entries:
        entry.overlap_label = int(km.labels_[index])

    print("CLUSTERING ROUND 2 COMPLETED, CLUSTER LABLES HAVE BEEN ASSSIGNED.")
    return num_of_labels


# here we craft our second set of clusters, built using clever reverse IDF on host-overlap frequency
# to hostname calls.
# [OPT] PLEASE FUCKING PARALLELIZE ME PLZ OH GOD PLZ [OPT]
# [IMP] Consider different domain levels, is xygfvf.com really != xygfvf.net
def hostBipartiteGraph(log_entries):
    print("\n\n┻━┻︵  \(°□°)/ ︵ ┻━┻ ")
    print("BEGIN CREATING HOST BIPARTITE GRPAH OF OVERLAPING CALLS TO A DOMAIN")

    # we keep a key value dict based on requested domains, the domain key corresponds to the last log
    # entry that records a host which asked for it.
    hosts = dict()
    for entry in log_entries:
        try:
            # first validate this is legitimate ipv4 structure.
            # socket.inet_aton(entry.src_ip)
            # now either create this entry or add to it.
            if entry.src_ip in hosts:
                hosts[entry.src_ip] += 1 # used to craft our list of hosts that query more than 1 domain.
            else:
                hosts[entry.src_ip] = 1
        except (socket.error, KeyError) as e:
            # print('address/netmask is invalid: %s' % entry.src_ip)
            print(e)
        
    # now generate our list of hosts that queried more than 1 domain.
    hosts_with_more_than_1_req = []
    for entry in log_entries:
        try:
            if hosts[entry.src_ip] > 1:
                hosts_with_more_than_1_req.append(entry)
        except KeyError:
            print("no entry: "+entry.src_ip)

    # [OPT]SHITTY N^2 solution, plz parralelize me or make me smarter
    # [+] You could sort host objects by domains calls, then use a binary search accross the hosts.
    # given both lists, parse each of the hosts for a given domain, keeping tabs on who called them.
    # This will build our per domain vectors that make up our matrix.
    print("Building the matrix")
    # this holds the numpy array representation of our matrix. We declare the full array
    # of zeros here as to take advantage of numpy fully by giving it contiguous memory 
    # to work with right from the start.
    matrix_M = np.zeros(shape=(len(log_entries), len(hosts_with_more_than_1_req)))
    # now we can finally itterate through each domain, building its vector and tacking it 
    # onto our matrix_M.
    matrix_M_index = 0
    count = 1
    for log_entry in log_entries:
        try:
            # armed with the log entry corresponding to the domain, cycle each host in an attempt to find
            # those that have called on it.
            count2 = 0
            for host_entry in hosts_with_more_than_1_req: 
                if host_entry.request_url == log_entry.request_url:
                    count2 += 1
                    # calculate the significance of this host having called this domain based on the # of 
                    # domains the host called in our period.
                    weighted_value = 1/int(hosts[log_entry.src_ip])
                    log_entry.overlap_vector.append(weighted_value)
                else:
                    count2 += 1
                    log_entry.overlap_vector.append(0)

            # ensure that there are no bad vectors, print out the culprits for validation
            if len(log_entry.overlap_vector) == len(hosts_with_more_than_1_req):
                matrix_M[matrix_M_index] = log_entry.overlap_vector
            else:
                print("BAD ENTRY IN THE MATRIX")
                print(log_entry.request_url)
                print(len(log_entry.overlap_vector))
                print(count)

            matrix_M_index += 1
            count += 1
        except KeyError:
            continue

    print("perform the PCA transform for dense vectors.")
    # compute the transposition of matrix_M
    pca = PCA()
    matrix_M_transform = pca.fit_transform(matrix_M)

    print("assign the lable to the log entry objects.")
    # for each log_entry, assign the dense vector we just calculated to the object.
    index = 0
    for log_entry in log_entries:
        log_entry.overlap_vector = matrix_M_transform[index]
        index += 1

    # important to return this value so we can 
    print("BIPARTITE GRAPH IS COMPLETED, DENSE FEATURE VECTORS ASSIGNED TO EACH LOG ENTRY")


# implementation of x-means clustering with the help of scikit.
# Q's: | what should our n_cluster size be? | how larger should batch_size be | what about other tweaks to k-means? |
def xmeansClusteringSubsets(nxkdomains, log_entries, nxkdomain_subset_size):
    print("\n\n┻━┻︵  \(°□°)/ ︵ ┻━┻ ")
    print("STARTING THE CLUSTERING OF THE STATISTICAL VECTORS.")
    # in order to ensure that we use numpy efficiently, first declare your full
    # array size, followed by assigning values into it. This ensures we follow 
    # a single contiguous block of memory.
    num_of_nxkdomain_subsets = len(nxkdomains)
    numpy_fv_array = np.zeros(shape=(num_of_nxkdomain_subsets, 32))

    # craft a master numpy array of all feature vectors. 
    index = 0
    for subset_object in nxkdomains:
        numpy_fv_array[index] = subset_object.feature_vector
        index += 1

    # build our kmeans object
    #km = MiniBatchKMeans(n_clusters=10, init='k-means++', max_iter=300, n_init=10, batch_size=10000)
    km = MiniBatchKMeans(init='k-means++', max_iter=300, batch_size=10000)

    # fit our feature vectors to lables.
    t0 = time.time()
    label_set = km.fit_predict(numpy_fv_array)
    num_of_labels = len(np.unique(label_set))
    print("done clustering in %0.3fs" % (time.time() - t0))

    # now apply those lables back into the log entry objects for the log entries contained in our subsets.
    # at this point, we no longer need the subsets, we will store which cluster the log entry belongs to with
    # the id of the cluster.
    size = len(numpy_fv_array)
    index = 0
    for x in range(0, size):
        for i in range(index, index+nxkdomain_subset_size):
            log_entries[i].statistical_cluster_label = km.labels_[x]
        index += nxkdomain_subset_size


    print("CLUSTER IDs HAVE BEEN ASSIGNED, ROUND 1 OF CLUSTERING COMPLETED.")
    return num_of_labels

#here we create our list of subsets (section 4.1.2)
def craftSubsets(master_list, subset_size):
    print("\n\n┻━┻︵  \(°□°)/ ︵ ┻━┻  ")
    print("CRAFTING SUBSETS NXk")

    #place holder for the list of subsets we are returning.
    nxk_subsets = []

    #check if your master list is divisible by the provided subset size, pad if needed
    list_length = len(master_list)
    print("LENGTH BEFORE PADDING: "+str(list_length))
    mod = list_length % subset_size 
    if(mod != 0):
        #gather the # of dummy log entries required to pad our list
        pad_size = subset_size - mod

        #append log entries to our list.
        dummy_log_entry = logEntry("dummy", "google.com")
        for x in range(0, pad_size):
            master_list.append(dummy_log_entry)


    #we need the size to know how long to loop
    length = len(master_list)
    print("LENGTH AFTER PADDING: "+str(length))
    num_of_subsets = int(length / subset_size)
    print("YOUR TOTAL # OF SUBSETS: "+str(num_of_subsets))

    #loop through the master list, crafting subsets.
    index = 0
    for i in range(0, num_of_subsets):
        subset = []
        for q in range(0, subset_size):
            subset.append(master_list[index])
            index+=1
        
        #make a new nxk_subset object and append it to our list of subsets
        nxk_subset = nxkDomainSubset(subset)
        nxk_subsets.append(nxk_subset)

    print("SUBSET GENERATION COMPLETE")
    return nxk_subsets



#this is the function for the file ingestion worker threads. 
def ingestLogFile():
    print("┻━┻︵  \(°□°)/ ︵ ┻━┻  ")
    print("BEGIN LOG INGESTION!!!")
    global master_log_entry_list

    #open file for reading.
    log_file = open("/home/x90/security/projects/dga_detection/sample_data/nov14th.csv")

    for line in log_file:
        #remove all the quotations from the logger csv
        line = line.replace("\"", "")
        line = line.replace("\'", "")
        parts = line.split(",")

        #parse the info from the line. (possible speed optimization with csv module)
        nxdomain = parts[0]
        source_address = parts[1]
        source_host = parts[2]

        #create struct 
        #if there is no way to ID the host, just throw out the record... 
        if(source_host != "" and source_host != " "):
            log_entry = logEntry(str(source_host), str(nxdomain))
        elif(source_address != ""):
            log_entry = logEntry(str(source_address), str(nxdomain))

        #append entry to the master list.
        master_log_entry_list.append(log_entry)


    #close file
    log_file.close()

    print("LOG INGESTION COMPLETED")
    #return the list we just created
    return master_log_entry_list



if __name__ == "__main__":
    master_log_entry_list = ingestLogFile()
    
    #perform the basic clustering based on string and format
    #create your list of size Q subsets
    nxk_subsets = craftSubsets(master_log_entry_list, 10)

    #with a full list of subsets, it is time to craft their feature vectors.
    print("\n\n┻━┻︵  \(°□°)/ ︵ ┻━┻ ")
    print("BEGIN BUILDING STATISTICAL SUBSET VECTORS")
    vector_creation.calculateNgramFrequencies(nxk_subsets)
    vector_creation.calculateEntropyFeatures(nxk_subsets)
    vector_creation.calculateStructuralFeatures(nxk_subsets)
    #for subset in nxk_subsets: 
        #print(subset.feature_vector)
    print("STATISTICAL VECTORS HAVE BEEN BUILT AND ASSINGED TO EACH SUBSET.")

    #now perform x-means clustering on the feature vectors you just created.
    num_of_stats_labels = xmeansClusteringSubsets(nxk_subsets, master_log_entry_list, 10)

    #perform the host based overlap clustering.
    hostBipartiteGraph(master_log_entry_list)

    #perform the xmeans clustering on the dense vectors you just stuck onto the log_entries
    num_of_matrix_labels = xmeansClusteringLogEntries(master_log_entry_list)

    #perform the cluster corrolation to produce your final set of noise reduced clusters.
    print(str(num_of_stats_labels))
    print(str(num_of_matrix_labels))
    clusters = clusterCorrelation(master_log_entry_list, num_of_stats_labels, num_of_matrix_labels, 3, 50)
    count = 0
    print("LOLOLOLOlolO: "+str(len(clusters)))
    for cluster in clusters:
        print("========================================================================================")
        print("THESE ARE THE DOMAINS IN SET #%s" % str(count))
        for log_entry in cluster:
            print(log_entry.request_url)

        print("\n\n")
        count +=1


    #begin classification
    #clusterClassification()
        ### ROUND 1 ###
        # start by building / training an ADT for each type of known DGA based malware.

        # for every cluster, generate subsets of size 10 and run their FV against the ADT of
        # each malware type. If the subset meets the threashold, append its entries to the 
        # list of candidates for that malware type.

        # once all clusters have had a chance to check their subsets and we have full lists of 
        # candidates, start the second round 


        ### ROUND 2 ###
        # in round 2, we run through each individual log entry and compare them to our ADT tree. 

        # if a particular entry meets the threashold, append it to our final list of log entries
        # for each DGA type

        # we have 2 rounds to increase the accuracy of our tagging a log entry belonging to a 
        # particular DGA.



    # PRINT OUT EACH OF THE LISTS OF LOG_ENTRIES PER DGA TYPE.

    # PRINT OUT A LIST OF THE DOMAINS IN THOSE LISTS THAT ARE ACTIVE.
