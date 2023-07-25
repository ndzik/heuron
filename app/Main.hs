{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Main where

import Digits

pathToMNISTLabel :: FilePath
pathToMNISTLabel = "./data/train-labels-idx1-ubyte"

pathToMNISTImage :: FilePath
pathToMNISTImage = "./data/train-images-idx3-ubyte"

main :: IO ()
main = do
  -- TODO: Create a Stream from Image and Label files.
  -- s <- Stream.chunk batchSize $ Stream.zip streamMNISTLabels streamMNISTImages
  -- Stream.fold (\ann (labels, images) -> do
  --                  (accuracy, ann) <- runTrainer (oneEpoch images labels) $ TrainerState ann CategoricalCrossEntropy
  --                  print accuracy
  --                  return ann
  --               ) ann s
  streamMNISTLabels pathToMNISTLabel
  return ()
