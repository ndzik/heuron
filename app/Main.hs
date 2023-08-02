{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Main where

import Control.Monad.Except (ExceptT, runExceptT)
import Data.Data (Proxy (..))
import Data.Functor ((<&>))
import Digits
import GHC.TypeLits (natVal)
import GHC.TypeNats (KnownNat)
import Heuron.Functions
import Heuron.V1
import Heuron.V1.Batched
import Linear.V
import Streaming (liftIO)
import qualified Streaming as S
import qualified Streaming.Prelude as S
import System.Random (getStdGen)

pathToMNISTLabel :: FilePath
pathToMNISTLabel = "./data/train-labels-idx1-ubyte"

pathToMNISTImage :: FilePath
pathToMNISTImage = "./data/train-images-idx3-ubyte"

main :: forall pixelCount batchSize numOfImages. (pixelCount ~ 784, batchSize ~ 100, numOfImages ~ 60000) => IO ()
main = do
  -- Describe network.
  rng <- getStdGen
  inputLayer <- Layer <$> randomM' @pixelCount @pixelCount rng <*> randomV' @pixelCount rng <*> return zero
  hiddenLayer00 <- Layer <$> randomM' @1024 @pixelCount rng <*> randomV' @1024 rng <*> return zero
  hiddenLayer01 <- Layer <$> randomM' @1024 @1024 rng <*> randomV' @1024 rng <*> return zero
  hiddenLayer02 <- Layer <$> randomM' @1024 @1024 rng <*> randomV' @1024 rng <*> return zero
  outputLayer <- Layer <$> randomM' @10 @1024 rng <*> randomV' @10 rng <*> return zero
  let ann =
        inputLayer ReLU (StochasticGradientDescent 1.0)
          :>: hiddenLayer00 ReLU (StochasticGradientDescent 1.0)
          :>: hiddenLayer01 ReLU (StochasticGradientDescent 1.0)
          :>: hiddenLayer02 ReLU (StochasticGradientDescent 1.0)
            =| outputLayer Softmax (StochasticGradientDescent 1.0)

  -- Train network.
  labels <- streamMNISTLabels pathToMNISTLabel
  imgs <- streamMNISTImages @pixelCount pathToMNISTImage
  let s = streamOfSize @batchSize $ S.zip labels imgs

  print "Starting training..."
  _res <-
    runExceptT $
      S.foldM_
        ( \ts (labels, images) -> do
            (accuracy, ts') <- runTrainer (oneEpoch images labels) ts
            liftIO $ print accuracy
            return ts'
        )
        (pure $ TrainerState ann CategoricalCrossEntropy)
        pure
        $ S.take 10 s
  print "Done training."
  return ()

streamOfSize ::
  forall b.
  (KnownNat b) =>
  S.Stream (S.Of (V 10 Double, V 784 Double)) (ExceptT HeuronError IO) () ->
  S.Stream (S.Of (Input b 10 Double, Input b 784 Double)) (ExceptT HeuronError IO) ()
streamOfSize s =
  let batchSize = fromIntegral $ natVal (Proxy @b)
      substreams = S.chunksOf batchSize s
   in S.mapsM
        ( \substream -> do
            labelsAndImages <- S.toList substream
            let collapsed = S.mapOf (S.bimap (mkV'' @b) (mkV'' @b) . unzip) labelsAndImages
            return collapsed
        )
        substreams
  where
    mkV'' :: forall b a. (KnownNat b) => [a] -> V b a
    mkV'' as = case mkV @b as of
      Nothing -> error "mkV''"
      Just v -> v
