{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

module Main where

import V2Example
import V2ExamplePytorch

main :: forall pixelCount batchSize numOfImages hiddenNeuronCount. (hiddenNeuronCount ~ 16, pixelCount ~ 784, batchSize ~ 100, numOfImages ~ 60000) => IO ()
main = do
  -- generateV2Pytorch
  executeV2Network @pixelCount @batchSize @numOfImages @hiddenNeuronCount
  print "Heuron Done"
