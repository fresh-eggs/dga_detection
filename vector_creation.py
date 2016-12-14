# below are a set of funtions that can be used to generate specific feature vectors,
# given a list of objects that contain strings.
import statistics
import math
from nltk.tokenize import regexp_tokenize
from nltk.util import ngrams
from nltk.probability import FreqDist


#build our feature vectors and stick them into each subset NXk
def calculateNgramFrequencies(nxk_subsets):
    #process each NXkDomain subset and build their stats based feature vectors.
    domain = []
    frequencies = []
    for subset_object in nxk_subsets:
        #gathe a list of unique characters in the subset of domains.
        char_list_tokenized = []
        # create a list of unique chars, tokenize per char.
        for log_entry in subset_object.log_entries:
            # covert our request URL into a set, this way we can join all the chars 
            # accross the domains in the subset. This creates a unique list of chars
            url_characters = list(log_entry.request_url)
            char_list_tokenized = list(set(char_list_tokenized+url_characters))


        #Create your n-grams
        one_gram = ngrams(char_list_tokenized, 1)
        two_gram = ngrams(char_list_tokenized, 2)
        three_gram = ngrams(char_list_tokenized, 3)
        four_gram = ngrams(char_list_tokenized, 4)

        #compute frequency distribution for all the n-grams
        one_freq_dist = FreqDist(one_gram)
        two_freq_dist = FreqDist(two_gram)
        three_freq_dist = FreqDist(three_gram)
        four_freq_dist = FreqDist(four_gram)

        #get the median, mode and avg for the list of frequencies.
        freq_distributions = [one_freq_dist, two_freq_dist, three_freq_dist, four_freq_dist]
        
        #now that we have the frequncy distributions for all n-grams, we need to calculate
        #the avg, median ans stddev of each of the sets. 
        #We then append these vlaues to the feature vector
        for freq_distribution in freq_distributions:
            # get a list of all our n-grams by name.
            keys = freq_distribution.hapaxes()
            list_of_frequencies = []

            #avg
            total = 0
            for i in range(0,len(keys)):
                # here we use the get function in freqdist ojects. 
                # For a key, it returns the frequency of the ngram.
                total += float(freq_distribution.get(keys[i]))
                #might as well make out list of freqs at the same time
                list_of_frequencies.append(float(freq_distribution.get(keys[i])))
            avg = total / len(keys)
            subset_object.feature_vector.append(float(avg))

            #median
            list_of_frequencies.sort()
            median = statistics.median(list_of_frequencies)
            subset_object.feature_vector.append(float(median))

            #standard deviation.
            stdev = statistics.stdev(list_of_frequencies)
            subset_object.feature_vector.append(float(stdev))



def calculateEntropyFeatures(nxk_subsets):
    #entropy features (shannon entropy)
    for subset_object in nxk_subsets:
        # calcualte the avg and stdev of the entropy levels of FQDN, 2LD and 3LD
        entropy_set_FQDN = []
        entropy_set_2LD = []
        entropy_set_3LD = []

        for log_entry in subset_object.log_entries:
            entry_failure = False
            #create our 2LD and 3LD substrings
            #  ************** CHECK THIS SPLIT
            try:
                split_results = log_entry.request_url.split(".")
                length = len(split_results)
                if(length > 2):
                    string_2LD = split_results[-2] #second to last
                    string_3LD = split_results[-3] #3rd to last
                elif(length == 2):
                    string_2LD = split_results[-2] #second to last
                    string_3LD = "" #3rd to last
                elif(length == 1):
                    string_2LD = "" #second to last
                    string_3LD = "" #3rd to last
            except:
                string_2LD = "" #second to last
                string_3LD = "" #3rd to last
                print("Log Entry Could not be broken into TLDs: "+log_entry.request_url)
            
            # get probability and entropy for the FQDN
            prob = [float(log_entry.request_url.count(c)) / len(log_entry.request_url) for c in dict.fromkeys(list(log_entry.request_url))]
            entropy = - sum([ p * math.log(p) / math.log(2.0) for p in prob ])
            # append it to our temp list of entropy per domain in the subset.
            entropy_set_FQDN.append(float(entropy))

            # get probability and entropy for the 2LD
            prob = [float(string_2LD.count(c)) / len(string_2LD) for c in dict.fromkeys(list(string_2LD))]
            entropy = - sum([ p * math.log(p) / math.log(2.0) for p in prob ])
            # append it to our temp list of entropy per domain in the subset.
            entropy_set_2LD.append(float(entropy))

            # get probability and entropy for the 3LD
            prob = [float(string_3LD.count(c)) / len(string_3LD) for c in dict.fromkeys(list(string_3LD))]
            entropy = - sum([ p * math.log(p) / math.log(2.0) for p in prob ])
            # append it to our temp list of entropy per domain in the subset.
            entropy_set_3LD.append(float(entropy))


        # append average of the entropy values 
        total = 0
        for i in range(0,len(entropy_set_FQDN)):
            total += entropy_set_FQDN[i]
        avg = total / len(entropy_set_FQDN)
        subset_object.feature_vector.append(float(avg))

        # append the std deviation of the entropy values
        entropy_set_FQDN.sort()
        stdev = statistics.stdev(entropy_set_FQDN)
        subset_object.feature_vector.append(float(stdev))

        # append average of the entropy values 
        for i in range(0,len(entropy_set_2LD)):
            total += entropy_set_2LD[i]
        avg = total / len(entropy_set_2LD)
        subset_object.feature_vector.append(float(avg))

        # append the std deviation of the entropy values
        entropy_set_2LD.sort()
        stdev = statistics.stdev(entropy_set_2LD)
        subset_object.feature_vector.append(float(stdev))

        # append average of the entropy values 
        for i in range(0,len(entropy_set_3LD)):
            total += entropy_set_3LD[i]
        avg = total / len(entropy_set_3LD)
        subset_object.feature_vector.append(float(avg))

        # append the std deviation of the entropy values
        entropy_set_3LD.sort()
        stdev = statistics.stdev(entropy_set_3LD)
        subset_object.feature_vector.append(float(stdev))


# calculate 14 unique features based on unique tlds, domain name length, etc...
def calculateStructuralFeatures(nxk_subsets):
    for subset_object in nxk_subsets:
        # calculate the avg, median, stdev and variance of the length of domains in NXk
        lengths = []
        for log_entry in subset_object.log_entries:
            lengths.append(float(len(log_entry.request_url)))

        #avg
        total = 0
        for i in range(0,len(lengths)):
            total += float(lengths[i])
        avg = total / len(lengths)
        subset_object.feature_vector.append(float(avg))

        #median
        lengths.sort()
        median = statistics.median(lengths)
        subset_object.feature_vector.append(float(median))

        #standard deviation.
        stdev = statistics.stdev(lengths)
        subset_object.feature_vector.append(float(stdev))

        #variance
        variance = statistics.variance(lengths)
        subset_object.feature_vector.append(float(variance))


        # get the number of domain levels per domain in NXk
        num_of_levels = []
        for log_entry in subset_object.log_entries:
            #quickly split the string on "." to generate a list of levels
            list_of_levels = log_entry.request_url.split(".")

            #now collect the # of levels.
            num_of_levels.append(len(list_of_levels))

        #avg
        total = 0
        for i in range(0,len(num_of_levels)):
            total += float(num_of_levels[i])
        avg = total / len(num_of_levels)
        subset_object.feature_vector.append(float(avg))

        #median
        num_of_levels.sort()
        median = statistics.median(num_of_levels)
        subset_object.feature_vector.append(float(median))

        #standard deviation.
        stdev = statistics.stdev(num_of_levels)
        subset_object.feature_vector.append(float(stdev))

        #variance
        variance = statistics.variance(num_of_levels)
        subset_object.feature_vector.append(float(variance))

        # here we want a count of the unique chars across our domains.
        # we are re-using the tokenization routine we made earlier.
        char_list = []
        for log_entry in subset_object.log_entries:
            url_characters = list(log_entry.request_url)
            char_list = list(set(char_list+url_characters))
        count = len(char_list)
        subset_object.feature_vector.append(float(count))

        # gather the number of unique TLDs through our subset of domains.
        unique_tlds = []

        # craft a set of all TLDs
        levels = []
        tlds = []
        for log_entry in subset_object.log_entries:
            levels = log_entry.request_url.split(".")
            tlds.append(levels[-1])
            unique_tlds = list(set(unique_tlds+tlds))
        subset_object.feature_vector.append(float(len(unique_tlds)))

        # gather the ratio of .com TLDs to everything else
        com_count = 0
        for log_entry in subset_object.log_entries:
            domain_levels = log_entry.request_url.split(".") 
            if "com" in domain_levels[-1]:
                com_count +=1
        com_ratio = float(com_count / len(subset_object.log_entries))
        subset_object.feature_vector.append(com_ratio)

        # get the avg, median and SD of the frequency dist. for the TLDs other than .com
        freq_dist = {}
        print("==============================================")
        for log_entry in subset_object.log_entries:
            print(log_entry.request_url)
            domain_levels = log_entry.request_url.split(".")
            # if it isnt .com, calculate the frequency.
            if "com" not in domain_levels[-1]:
                if domain_levels[-1] not in freq_dist:
                    freq_dist[domain_levels[-1]] = 1
                else:
                    freq_dist[domain_levels[-1]] += 1
        distribution = []            
        for values in freq_dist.values():
            print(str(values))
            distribution.append(values)

        #avg
        total = 0
        if(len(distribution) > 1):
            for value in distribution:
                total += float(value)
            avg = total / len(distribution)
            subset_object.feature_vector.append(float(avg))
        else:
            subset_object.feature_vector.append(0.0)

        #median
        #just keep a standard value for cases where there was only one TLD different from com
        if(len(distribution) > 1):
            distribution.sort()
            median = statistics.median(distribution)
            subset_object.feature_vector.append(float(median))
        else:
            subset_object.feature_vector.append(0.0)

        #standard deviation.
        if(len(distribution) > 1):
            stdev = statistics.stdev(distribution)
            subset_object.feature_vector.append(float(stdev))
        else:
            subset_object.feature_vector.append(0.0)

