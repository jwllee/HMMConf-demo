# HMMConf-demo
### An online conformance checking based on Hidden Markov Models (HMM)
This is a demo of the online conformance checking technique (HMMConf). The technique is based on Hidden Markov Models (HMM) and performs conformance checking by alternating between orienting the running case within the process and conformance computation. This tackles the challenge of balancing between making sense at the process level as the case reaches completion and putting emphasis on the current information at the same time.

The demo illustrates how the technique would work at an online scenario where events related to different cases are coming and conformance is being computed as they come in. The dataset is a real-life dataset of a [hospital billing process](https://doi.org/10.4121/uuid:76c46b83-c930-4798-a1c9-4be94dfeb741).

For more detail and the implementation, please check out the [Github repository](https://github.com/jwllee/HMMConf).
