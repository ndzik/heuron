module Main where

import Heuron.Test.V1.Batched.ActivationSpec
import Heuron.Test.V1.Batched.BatchedSpec
import Heuron.Test.V1.Batched.TrainerSpec

main :: IO ()
main = do
  let tests =
        [ batchedSpec,
          activationSpec,
          trainerSpec
        ]
  sequence_ tests
