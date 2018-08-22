## Thoughts on keras backend implementation
### Python Wrapping
To function as an actual backend for keras LBANN needs to be callable from python. In this branch I was testing ways to make this happen. I found pybind11 in onnx documnetation, and swig from Tensorflow's python implementation. In general I found pybind to be significantly easier, and require less template type code. Also one incredibly motivating factor in this decision was that I couldn't for the life of me get swig to compile. Regardless of how this is accomplished, wrapping the cpp code with some python will be absolutely necessary for this task.
### What to wrap
I fumbled around with this for a bit. I feel like you probably want to wrap as little of lbann as necessary for this (unless you ever wanted LBANN to be fully importable in python ala tensorflow). 

The initial plan was to just create model parameters (layers etc) by wrapping the factory functions and put together a model like that, stubbing out the rest of the backend file. This ran into a major issue because of how keras and its backend actually interact. I came to find that to ever even get to the layer functions in this backend file you would need to implement many many building block functions which required hooking pretty deep into LBANN's internals. 

The result of this discovery can be seen in some of the hacking around I was doing in the weights class. I saw in the comments of this class that our weights resembled Tensorflow's variables. Tensorflow variables are used as tensors in keras. Basically all of the building block functions a keras backend requires is built using these tensor objects. Almost every function takes a tensor in, and returns a tensor. LBANN's weight class doesnt really fit exactly with the functionality keras needs, but it was more proof of concept work for me. 

I have thought more about this problem since I put this work on hold, and was thinking that it may make sense to even hook directly into hydrogen for LBANN's tensor representation. Getting this problem figured out and beginning to flesh out these lower level backend functions required by a keras backend was going to be where I started when returning to this task. 

### The resulting converter tool/lessons learned
I tried a hack which tacked on lbann protobuf model creation to functions that were called during the layer construction call from keras. This implementation had a whole slew of issues, and (in my opinion) would never work properly even if I kept hacking away on it. The backend file is far less aware of the network structure (etc) that we need to construct an LBANN protobuf model, and in general didn't expose the necessary information to complete this task. So I moved instead to write a standalone converter that just asks users to call the converter function with the model object they define in keras. 

There wasn't really a half measure available in regards to LBANN as a keras backend. I believe my assessment is correct that the options for keras-lbann integration are either convert keras to prototext (what I implemented) or fully flesh out a LBANN backend implementation for keras.
