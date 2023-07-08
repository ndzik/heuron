module Heuron.Test.V1.Batched.ActivationSpec where

import Heuron.V1 (mkM')
import Heuron.V1.Batched (Differentiable (derivative), Softmax (Softmax))

activationSpec :: IO ()
activationSpec = do
  let tests =
        [ softmaxSpec
        ]
  sequence_ tests

softmaxSpec :: IO ()
softmaxSpec = do
  samples <- mkM' @1 @3 [[0.7, 0.1, 0.2]]
  let result = derivative (Softmax @3) samples samples
  print result
