#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/utils/random.hpp"
namespace lbann {

lbann_comm* lbannpy_init(int rand_seed) {
  El::Initialize();
  //MPI_Init(NULL,NULL);
  auto* comm = new lbann_comm(0);
  init_random(rand_seed);
  init_data_seq_random(rand_seed);
  return comm;
}

}
