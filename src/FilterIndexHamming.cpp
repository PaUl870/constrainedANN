#include <iostream>
#include <vector>
#include <algorithm>
#include <string> 
#include <map>
#include <random>
#include <sys/time.h>
#include <sys/stat.h>
#include <chrono>

#include "readfile.h"
#include "utils.h"
#include "FilterIndex.h"
#include <faiss/Clustering.h> 
#include <faiss/IndexFlat.h>
#include <bits/stdc++.h>

using namespace std;

template <typename S>
// operator to print vectors, matrix
ostream& operator<<(ostream& os, const vector<S>& vector){
    // Printing vector elements using <<
    for (auto element : vector) {
        os << element << " ";
    }
    return os;
}

template <typename S>
ostream& operator<<(ostream& os, const vector<vector<S>>& matrix){
    // Printing matrix elements using <<
    for (const vector<S>& vector : matrix) {
        for (auto element : vector) {
            os << element << " ";
        }
    os << endl;
    }
    return os;
}

FilterIndex::FilterIndex(float* data, size_t d_, size_t nb_, size_t nc_, vector<vector<string>>properties_){
    dataset = data; // data
    d =d_; // dim
    nb = nb_; //num data points
    nc = nc_; // num clusters
    treelen = 4; //length of hamming tree, num miniclusters= treelen+1

    properties.resize(nb);
    uint16_t cnt=0;
    for (int i=0; i<nb; i++){
        for (string prp: properties_[i]){
            if (prLook.count(prp)==0){
                prLook[prp]=cnt;
                cnt++;
            }
            properties[i].push_back(prLook[prp]);
        }
        // beware:: sorting the properties will loose the position information!!
        // sort(properties[i].begin(), properties[i].end()); 
    }   
    cout<<cnt<<" total unique constraints"<<endl; 
    // Properties to location map
    numAttr = properties[0].size();
    for (int i=0; i<nb; i++){
        for (int j=0; j<numAttr; j++){
            PrpAtrMap[properties[i][j]] = j;
        }
    }
}

//NN index
// TODO: have more partitioning methods. Include BLISS
void FilterIndex::get_kmeans_index(string metric, string indexpath){
    centroids = new float[d*nc]; //provide random nc vectors
    cen_norms = new float[nc]{0};
    data_norms = new float[nb]{0};
    Lookup= new uint32_t[nb];
    counts = new uint32_t[nc+1]{0};
    //init: take uniform random points from the cluster
    int v[nb];
    randomShuffle(v , 0, nb);
    faiss::Clustering clus(d, nc);
    clus.centroids.resize(d*nc);
    for(uint32_t i = 0; i < nc; ++i) { 
        for(uint32_t j = 0; j < d; ++j){
            clus.centroids[i*d+j] = dataset[v[i]*d +j];
        }
        // memcpy(clus.centroids + i*d, dataset +v[i]*d, sizeof(*centroids) * d);
    }
    clus.verbose = d * nb * nc > (1L << 30);
    // display logs if > 1Gflop per iteration

    faiss::IndexFlatL2 index(d);
    clus.train(nb, dataset, index);
    memcpy(centroids, clus.centroids.data(), sizeof(*centroids) * d * nc);
    cout<<"centroids size: "<<clus.centroids.size()<<endl; //centroids (nc * d) if centroids are set on input to train, they will be used as initialization
    
    // if L2 get norms as well
    for(uint32_t j = 0; j < nc; ++j){ 
        cen_norms[j]=0; 
        for(uint32_t k = 0; k < d; ++k) {                 
            cen_norms[j] += centroids[j*d +k]*centroids[j*d +k];        
        } 
        cen_norms[j] = cen_norms[j]/2;
    }
    //centroids (nc * d) if centroids are set on input to train, they will be used as initialization
    for(uint32_t j = 0; j < nb; ++j){  
        for(uint32_t k = 0; k < d; ++k) {    
            data_norms[j]=0;             
            data_norms[j] += dataset[j*d +k]*dataset[j*d +k];        
        } 
        data_norms[j]=data_norms[j]/2;
    }
    //observation: clusters are not balanced if not initiliased with random vectors

     uint32_t* invLookup = new uint32_t[nb];
    // Lookup= new uint32_t[nb];
    // counts = new uint32_t[nc+1]{0};
    // this needs to be integrated in clustering, use openmp, use Strassen algorithm for matmul*
    //get best score cluster
    #pragma omp parallel for  
    for(uint32_t i = 0; i < nb; ++i) {  
        float bin, minscore, temp;
        minscore = 1000000;      
        for(uint32_t j = 0; j < nc; ++j){  
            temp =0;
            for(uint32_t k = 0; k < d; ++k) {                 
                temp += pow(dataset[i*d + k] - centroids[j*d +k], 2);        
            } 
            if (temp<minscore) {
                minscore=temp;
                bin = j;}    
        }
        invLookup[i] = bin;    
    }

     for(uint32_t i = 0; i < nb; ++i) {
        counts[invLookup[i]+1] = counts[invLookup[i]+1]+1; // 0 5 4 6 3
    }
    for(uint32_t j = 1; j < nc+1; ++j) {
        counts[j] = counts[j]+ counts[j-1]; //cumsum 
    }
    
    //argsort invLookup to get the Lookup
    iota(Lookup, Lookup+nb, 0);
    stable_sort(Lookup, Lookup+nb, [&invLookup](size_t i1, size_t i2) {return invLookup[i1] < invLookup[i2];});
    get_mc_propertiesIndex(); // this will change counts, Lookup; and add maxMC tree

    //save index files
    mkdir(indexpath.c_str(), 0777);
    FILE* f1 = fopen((indexpath+"/centroids.bin").c_str(), "w");
    fwrite(centroids, sizeof(float), nc*d, f1);
    FILE* f2 = fopen((indexpath+"/centroidsNorms.bin").c_str(), "w");
    fwrite(cen_norms, sizeof(float), nc, f2);
    FILE* f3 = fopen((indexpath+"/dataNorms.bin").c_str(), "w");
    fwrite(data_norms, sizeof(float), nb, f3);
    FILE* f4 = fopen((indexpath+"/Lookup.bin").c_str(), "w");
    fwrite(Lookup, sizeof(uint32_t), nb, f4);
    FILE* f5 = fopen((indexpath+"/counts.bin").c_str(), "w");
    fwrite(counts, sizeof(uint32_t), nc*(treelen+1)+1, f5);
    FILE* f6 = fopen((indexpath+"/maxMC.bin").c_str(), "w");
    fwrite(maxMC, sizeof(uint16_t), nc*(treelen+1)*3, f6);
}   

void FilterIndex::get_mc_propertiesIndex(){
    vector<vector<uint32_t>> maxMCIDs; //nested array to store the mini-clusters, change to uint32_t array later
    maxMCIDs.resize((treelen+1)*nc);
    maxMC = new uint16_t[3*(treelen+1)*nc]; 

    for (int clID = 0; clID < nc; clID++){
        if (counts[clID+1]- counts[clID]==0) continue;
        //get count of vector properties        
        //get the max
        for (int h=0; h<treelen; h++){ //iterate over tree height
            unordered_map<uint16_t, int> freq; //property to frequency map
            for (int i =counts[clID]; i< counts[clID+1]; i++){ // for all points in the cluster
                for (uint16_t x: properties[Lookup[i]]){
                    if(not_in(x, maxMC + (treelen+1)*clID*3, h)) freq[x]++;
                }
            }

            //choose property with max freq
            auto maxElement = max_element(freq.begin(), freq.end(),
                [](const pair<uint16_t, int>& p1, const pair<uint16_t, int>& p2) { return p1.second < p2.second;}
            );
            int r = (treelen+1)*3*clID + 3*h;
            maxMC[r+0]= PrpAtrMap[maxElement->first]; // property location
            maxMC[r+1]= maxElement->first; // property
            maxMC[r+2]= maxElement->second; // frequency, do we need this in maxMC??
        }

        //maxMC serves as a node list and node data size in hamming tree, where
        //node: property from maxMC
        //node data: corresponding vector IDs
        for (int i =counts[clID]; i< counts[clID+1]; i++){
            for (int j=0; j< treelen; j++){
                int r = (treelen+1)*3*clID + 3*j;
                // cout<<properties[Lookup[i]]<<" "<<maxMC[r]<<endl;
                if(properties[Lookup[i]][maxMC[r]]==maxMC[r+1]){ 
                    maxMCIDs[(treelen+1)*clID +j].push_back(Lookup[i]);
                    goto m_label;
                }
            }
            maxMCIDs[(treelen+1)*clID +treelen].push_back(Lookup[i]);
            m_label:;
        }
    }
    //need some assert statements
    //update Lookup, counts. Flatten the maxMCIDs into Lookup
    //each cluster now spans treelen+1 buckets
    Lookup= new uint32_t[nb];
    counts = new uint32_t[nc*(treelen+1)+1]{0}; 
    for (int clID = 0; clID < nc; clID++){
        for (int j=0; j< treelen+1; j++){
            int id = clID*(treelen+1) +j;
            counts[id+1] = counts[id]+ maxMCIDs[id].size();
            memcpy(Lookup+ counts[id], maxMCIDs[id].data(), sizeof(*Lookup) * maxMCIDs[id].size());
        }
    }
}

void FilterIndex::loadIndex(string indexpath){
    centroids = new float[d*nc]; 
    cen_norms = new float[nc]{0};
    data_norms = new float[nb]{0};
    Lookup= new uint32_t[nb];
    counts = new uint32_t[nc+1]; 
    // counts = new uint32_t[nc*(treelen+1)+1]; 
    // maxMC = new uint16_t[3*(treelen+1)*nc]; 

    FILE* f1 = fopen((indexpath+"/centroids.bin").c_str(), "r");
    fread(centroids, sizeof(float), nc*d, f1);       
    FILE* f2 = fopen((indexpath+"/centroidsNorms.bin").c_str(), "r");
    fread(cen_norms, sizeof(float), nc, f2);
    FILE* f3 = fopen((indexpath+"/dataNorms.bin").c_str(), "r");
    fread(data_norms, sizeof(float), nb, f3);
    FILE* f4 = fopen((indexpath+"/Lookup.bin").c_str(), "r");
    fread(Lookup, sizeof(uint32_t), nb, f4);
    FILE* f5 = fopen((indexpath+"/counts.bin").c_str(), "r");
    fread(counts, sizeof(uint32_t), nc+1, f5);

    // FILE* f5 = fopen((indexpath+"/counts.bin").c_str(), "r");
    // fread(counts, sizeof(uint32_t), nc*(treelen+1)+1, f5);
    // FILE* f6 = fopen((indexpath+"/maxMC.bin").c_str(), "r");
    // fwrite(maxMC, sizeof(uint16_t), nc*(treelen+1)*3, f6);
    get_mc_propertiesIndex();

    //this changes Lookup
    for (int i =0; i< nc*(treelen+1); i++){
        int m1 = counts[i];
        int m2 = counts[i+1];
        sort(Lookup+m1, Lookup+m2,
            [&](uint32_t a, uint32_t b) {
            return properties[a] < properties[b];
            });
    }
    // reorder data and index
    dataset_reordered = new float[nb*d];
    data_norms_reordered = new float[nb];
    properties_reordered = new uint16_t[nb*numAttr];
    for(uint32_t i = 0; i < nb; ++i) {
        copy(dataset+Lookup[i]*d, dataset+(Lookup[i]+1)*d , dataset_reordered+i*d);
        data_norms_reordered[i] = data_norms[Lookup[i]];
        memcpy(properties_reordered+ i*numAttr, properties[Lookup[i]].data(), sizeof(*properties_reordered) * numAttr);
    }
    // cout<<properties_reordered;
    delete dataset;
}

void FilterIndex::query(float* queryset, int nq, vector<vector<string>> queryprops, int num_results, int max_num_distances){
    neighbor_set = new int32_t[nq*num_results]{-1};
    cout<<"num queries: "<<nq<<endl;
    for (size_t i = 0; i < nq; i++){
        findNearestNeighbor(queryset+(i*d), queryprops[i], num_results, max_num_distances, i);
    }
}

// start from best cluster -> choose minicluster -> bruteforce search
void FilterIndex::findNearestNeighbor(float* query, vector<string> Stprops, int num_results, int max_num_distances, size_t qnum)
{   
    chrono::time_point<chrono::high_resolution_clock> t1, t2, t3, t4, t5, t6;
    vector<float> topkDist;
    vector<uint16_t> props;
    for (string stprp: Stprops){
        props.push_back(prLook[stprp]);
    }
    // sort(props.begin(), props.end());

    priority_queue<pair<float, uint32_t> > pq;
    float simv[nc];
    uint32_t simid[nc];
    for (uint32_t id=0; id<nc; id++){
        simv[id] = L2SqrSIMD16ExtAVX(query, centroids+id*d, cen_norms[id], d);
    }
    // need argsort here
    iota(simid, simid+nc, 0);
    stable_sort(simid, simid+nc, [&simv](size_t i1, size_t i2) {return simv[i1] > simv[i2];});

    priority_queue<pair<float, uint32_t> > Candidates_pq;
    uint32_t Candidates[max_num_distances];
    float score[max_num_distances];
    int seen=0, seenbin=0;

    float sim;
    float a=0,b=0;
    // t1 = chrono::high_resolution_clock::now();
    while(seen<max_num_distances && seenbin<nc){ 
        uint32_t bin = simid[seenbin];
        seenbin++;
        int id = bin*(treelen+1);
        bin = bin*(treelen+1);
        //get which disjoint part of the cluster query belongs to
        uint16_t membership = getclusterPart(maxMC+ bin*3 , props, treelen);
        bin = bin+membership;
        for (int i =counts[bin]; i< counts[bin+1]; i++){
            // __builtin_prefetch (properties_reordered +(i+2)*numAttr, 0, 2);
            //check if constraint statisfies
            int j =0;
            while (j<numAttr && properties_reordered[i*numAttr +j]== props[j]) j++;
            if (j==numAttr){
                Candidates[seen]=i; 
                seen++;
            }
        }
    }
    // t2 = chrono::high_resolution_clock::now();
    if (seen<num_results+1){
        for (int i =0; i< seen; i++){ 
            neighbor_set[qnum*num_results+ i] = Lookup[Candidates[i]];
        }
    }
    else{
        for (int i =0; i< seen; i++){ 
            //  __builtin_prefetch (dataset_reordered +Candidates[i+1]*d, 0, 2);
            score[i] = L2SqrSIMD16ExtAVX(query, dataset_reordered +Candidates[i]*d, data_norms_reordered[Candidates[i]], d);
            Candidates_pq.push({score[i], Lookup[Candidates[i]]});
        }
        for (int i =0; i< min(seen,num_results); i++){ 
            neighbor_set[qnum*num_results+ i] = Candidates_pq.top().second;
            Candidates_pq.pop();
        }
    }
    // cout<<"time: "<<chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count()<<" ";
}