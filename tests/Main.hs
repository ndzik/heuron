module Main where

import Heuron.Test.V1.Batched.ActivationSpec
import Heuron.Test.V1.Batched.BatchedSpec

main :: IO ()
main = do
  let tests =
        [ batchedSpec,
          activationSpec
        ]
  sequence_ tests
