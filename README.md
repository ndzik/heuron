# Heuron

This is a prototype implementation for describing neural networks in Haskell.
The basic idea is to have a backend agnostic DSL, which can have FPGAs, GPUs or CPUs as a target.

## Heuron.V1

V1 is an experiment of how I can achieve my goal of **correct by construction** neural networks.
For the meantime, V1 is a purely CPU based implementation using 100% Haskell code.

  * Heuron.V1.Single:
    - Initial API for describing a feed-forward neural net without backpropagation primitives
      on singular observations, i.e. batch size of 1. This will ultimately be removed together
      with all of V1, when V2 is realized.
  * Heuron.V1.Batched:
    - This contains a feed-forward neural net with training capabilities using backpropagation.
      The theory of how to realize and implement a neural net is not difficult, the complex
      part was how I could lift most of the construction into the type-level s.t. the compiler
      has all information available to prohibit the user from describing NNs that are:
        * unsupported
        * contain incompatible layers:
          * This includes forward & backward pass separately

## Heuron.V2

With my experience from implementing V1 I want to generalize the created interfaces and make
them abstract enough to allow different net-generation backends. E.g. it should be possible
to let this library generate a GPU optimized neural net for training and a CPU/FPGA targeted
software net for execution. All with the same code.

## FAQ

**Q:** Why do you do this if there are things like TensorFlow, PyTorch, etc.?

**A:** I like types, I like proofs, I like static analysis. I like to learn stuff and build
       things from the ground up. I do, because I am.

**Q:** Is this possible with Haskell?

**A:** We will see.
