#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include <pybind11/complex.h>

#include <lbann/proto/factories.hpp>
//#include <lbann/comm.hpp>
#include <lbann/base.hpp>
#include <lbann/weights/weights.hpp>
namespace py = pybind11;


PYBIND11_MODULE(lbann_backend, m) {
  m.doc() = "LBANN model constructors"; // optional module docstring
  
  //Model Constructor
  m.def("construct_model", &lbann::proto::construct_model, "Construct LBANN model");
 
  //Layer Graph Constructor
  m.def("construct_layer_graph", &lbann::proto::construct_layer_graph, "Instantiate layer graph");

  //Layer Constructor
  m.def("construct_layer_data_cpu", &lbann::proto::construct_layer<data_layout::DATA_PARALLEL, El::Device::CPU>, "Instantiate data parallel cpu layer");
  
  m.def("construct_layer_model_cpu", &lbann::proto::construct_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>, "Instanriate model parallel cpu layer");
  #ifdef LBANN_HAS_CUDA
  m.def("construct_layer_data_gpu", &lbann::proto::construct_layer<data_layout::DATA_PARALLEL, El::Device::GPU>, "Instantiate data parallel gpu layer");

  m.def("construct_layer_model_gpu", &lbann::proto::construct_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>, "Instantiate model parallel gpu layer");
  #endif
  //Objective Function constructor
  m.def("construct_objective_function", &lbann::proto::construct_objective_function, "Instantiate objective function"); 
  
  //Metric Constructor
  m.def("construct_metric", &lbann::proto::construct_metric, "Instantiate metric function");

  //Optimizer Constructor
  m.def("construct_optimizer", &lbann::proto::construct_optimizer, "Instantiate optimizer");

  //Weights Constructor
  m.def("construct_weights", &lbann::proto::construct_weights, "Instantiate weights");

  //Callback Constructor
  m.def("construct_callback", &lbann::proto::construct_callback, "Instantiate callbacks");

  py::class_<lbann::lbann_comm> lbann_comm(m, "lbann_comm");
  lbann_comm
      .def(py::init<>());
  
  
  py::class_<lbann::weights> weights(m, "weights");
  weights
      .def(py::init<lbann::lbann_comm*>())
      .def("set_name", &lbann::weights::set_name)
      .def("get_name", &lbann::weights::get_name)
      .def("setup", (void (lbann::weights::*)(std::vector<int>)) &lbann::weights::setup)
      .def("get_dims", &lbann::weights::get_dims)
      .def("set_value", (void (lbann::weights::*)(lbann::DataType value, std::vector<int>)) &lbann::weights::set_value)
      .def("get_values", &lbann::weights::get_values_pybind); 
  
      //.def_property("m_name", &lbann::weights::get_name, &lbann::weights::set_name);


  m.def("pyban_init", &lbann::lbannpy_init ,"Initialize communicator");
}

