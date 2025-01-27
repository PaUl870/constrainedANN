#include <iostream>
#include <fstream>
#include "FilterIndex.h"

#include <unistd.h>


void peak_memory_footprint()
{

  unsigned iPid = (unsigned)getpid();

  std::cout << "PID: " << iPid << std::endl;

  std::string status_file = "/proc/" + std::to_string(iPid) + "/status";
  std::ifstream info(status_file);
  if (!info.is_open())
  {
    std::cout << "memory information open error!" << std::endl;
  }
  std::string tmp;
  while (getline(info, tmp))
  {
    if (tmp.find("Name:") != std::string::npos || tmp.find("VmPeak:") != std::string::npos || tmp.find("VmHWM:") != std::string::npos)
      std::cout << tmp << std::endl;
  }
  info.close();
}

int main(int argc, char** argv)
{
    //default
    string metric = "L2";
    int mode = 0;
    string algo ="kmeans";
    size_t nc =0;
    // size_t buffer_size =0;
    size_t nprobe =0;

    size_t d, nb, nq, num_results; 
    string datapath, Attripath, querypath, queryAttripath, indexpath, GTpath;
    int success = argparser(argc, argv, &datapath, &Attripath, &querypath, &queryAttripath, &indexpath, &GTpath, &nc, &algo, &mode, &nprobe);

    float* data = fvecs_read(datapath.c_str(), &d, &nb);
    vector<vector<string>> properties = getproperties(Attripath,' ');
    // nc = atoi(argv[2]); // num clusters
    FilterIndex myFilterIndex(data, d, nb, nc, properties, algo, mode);
    myFilterIndex.loadIndex(indexpath);
    cout << "Loaded" << endl;


    float* queryset = fvecs_read(querypath.c_str(), &d, &nq);
    vector<vector<string>> queryprops = getproperties(queryAttripath,' ');
    int* queryGTlabel = ivecs_read(GTpath.c_str(), &num_results, &nq);
    cout << "Query files read..." << endl;

    chrono::time_point<chrono::high_resolution_clock> t1, t2;
    t1 = chrono::high_resolution_clock::now();
    myFilterIndex.query(queryset, nq, queryprops, num_results, nprobe);
    t2 = chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = t2 - t1;
    peak_memory_footprint();

    int32_t* output = myFilterIndex.neighbor_set;
    int output_[num_results*nq];
    copy(output, output+num_results*nq , output_);
    cout<<"numClusters, buffersize, QPS, Recall100@100 :"<<endl;
    //QPS and recall
    double QPS;
    double recall = RecallAtK(queryGTlabel, output_, num_results, nq);
    printf("%d,%d,%f,%f\n",nc, nprobe, nq/diff.count(), recall);
}

