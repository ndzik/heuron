{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Heuron.Test.V1.Batched.BatchedSpec where

import Data.Functor ((<&>))
import Heuron.V1
import Heuron.V1.Batched

batchedSpec :: IO ()
batchedSpec = do
  inputLayer <- Layer <$> mkM' @3 @6 [[1 .. 6], [1 .. 6], [1 .. 6]] <*> mkV' @3 [1, 2, 3]
  hiddenLayer <- Layer <$> mkM' @3 @3 [(+ 1) <$> [1 .. 3] | _ <- [1 .. 3]] <*> mkV' @3 [1, 2, 3]
  outputLayer <- Layer <$> mkM' @2 @3 [(+ 1) <$> [1 .. 3] | _ <- [1 .. 2]] <*> mkV' @2 [1, 2]
  let ann = inputLayer id :>: hiddenLayer id =| outputLayer id
      ann' = reverseNetwork ann
  -- Batchsize of 4 with 6 inputs.
  batchedInputs <- mkM' @4 @6 [[1 .. 6] | _ <- [1 .. 4]]
  let forwardResult = forward ann batchedInputs
  print "batchedSpec"
