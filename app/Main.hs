{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import V2Example

main :: forall pixelCount batchSize numOfImages hiddenNeuronCount. (hiddenNeuronCount ~ 16, pixelCount ~ 784, batchSize ~ 100, numOfImages ~ 57321) => IO ()
main = executeV2Network @pixelCount @batchSize @numOfImages @hiddenNeuronCount
