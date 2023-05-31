module Main where

import Heuron.Test.V1.Batched.BatchedSpec

main :: IO ()
main = do
  let tests = [batchedSpec]
  sequence_ tests
