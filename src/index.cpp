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
    string basepath, labelpath, indexpath;
    //default
    string metric = "L2";
    int mode = 1;
    string algo ="kmeans";
    size_t nc =1024;

    int success = argparser(argc, argv, &basepath, &labelpath, &indexpath, &nc, &algo, &mode);

    size_t d, nb; 
    float* data = fvecs_read(basepath.c_str(), &d, &nb);
    vector<vector<string>> properties = getproperties(labelpath,' ');
    cout << "Data files read" << endl;
    chrono::time_point<chrono::high_resolution_clock> t1, t2;

    t1 = chrono::high_resolution_clock::now();
    FilterIndex myFilterIndex(data, d, nb, nc, properties, algo, mode);
    myFilterIndex.get_index(metric, indexpath, mode);
    t2 = chrono::high_resolution_clock::now();
    peak_memory_footprint();

    cout<<"Index time: "<<chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count()<<endl;
    // cout << "Indexed at: " << indexpath << endl;

    return 0;
}
