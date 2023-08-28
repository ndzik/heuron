module Main where

import Heuron.Test.V1.Batched.ActivationSpec
import Heuron.Test.V1.Batched.BatchedSpec
import Heuron.Test.V1.Batched.LossSpec
import Heuron.Test.V1.Batched.TrainerSpec
import Heuron.Test.V2.NetworkSpec

main :: IO ()
main = do
  let tests =
        [ batchedSpec,
          activationSpec,
          trainerSpec,
          lossSpec,
          compileTimeCheck
        ]
  sequence_ tests
