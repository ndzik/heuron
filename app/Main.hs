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

import Control.Lens
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
import Text.Printf (printf)

pathToMNISTLabel :: FilePath
pathToMNISTLabel = "./data/train-labels-idx1-ubyte"

pathToMNISTImage :: FilePath
pathToMNISTImage = "./data/train-images-idx3-ubyte"

scaledBy :: (KnownNat n, KnownNat m) => Double -> Input n m Double -> Input n m Double
scaledBy x = fmap (fmap (* x))

main :: forall pixelCount batchSize numOfImages hiddenNeuronCount. (hiddenNeuronCount ~ 32, pixelCount ~ 784, batchSize ~ 100, numOfImages ~ 60000) => IO ()
main = do
  -- Describe network.
  rng <- getStdGen
  inputLayer <- Layer <$> (scaledBy (1 / 784) <$> randomM' @hiddenNeuronCount @pixelCount rng) <*> randomV' @hiddenNeuronCount rng <*> return zero
  hiddenLayer00 <- Layer <$> (scaledBy (1 / 32) <$> randomM' @hiddenNeuronCount @hiddenNeuronCount rng) <*> randomV' @hiddenNeuronCount rng <*> return zero
  hiddenLayer01 <- Layer <$> (scaledBy (1 / 32) <$> randomM' @hiddenNeuronCount @hiddenNeuronCount rng) <*> randomV' @hiddenNeuronCount rng <*> return zero
  outputLayer <- Layer <$> randomM' @10 @hiddenNeuronCount rng <*> randomV' @10 rng <*> return zero
  let learningRate = 0.001
      ann =
        inputLayer ReLU (StochasticGradientDescent learningRate)
          :>: hiddenLayer00 ReLU (StochasticGradientDescent learningRate)
          :>: hiddenLayer01 ReLU (StochasticGradientDescent learningRate)
            =| outputLayer Softmax (StochasticGradientDescent learningRate)
      initialTrainerState = TrainerState ann CategoricalCrossEntropy

  -- Train network.
  labels <- streamMNISTLabels pathToMNISTLabel
  imgs <- streamMNISTImages @pixelCount pathToMNISTImage
  let onlyThree (label, _) = label ^?! ix 2 == 1
      s = streamOfSize @batchSize $ S.zip labels imgs

  print "Starting training..."
  _res <- runExceptT $ S.foldM_ trainNetwork (pure initialTrainerState) pure s
  print "Done training."
  where
    forwardNetwork ts (labels, images) = do
      (prediction, ts') <- liftIO $ runTrainer (trainForward images) ts
      liftIO . putStrLn $ printf "Network:\n%s" (show $ _network ts')
      liftIO . putStrLn $ printf "Prediction: %s" (show prediction)
      return ts'
    trainNetwork ts (labels, images) = do
      (accuracy, ts') <- liftIO $ runTrainer (oneEpoch images labels) ts
      liftIO . putStrLn $ printf "Accuracy: %.2f" accuracy
      return ts'

streamOfSizeFiltered ::
  forall b.
  (KnownNat b) =>
  ((V 10 Double, V 784 Double) -> Bool) ->
  S.Stream (S.Of (V 10 Double, V 784 Double)) (ExceptT HeuronError IO) () ->
  S.Stream (S.Of (Input b 10 Double, Input b 784 Double)) (ExceptT HeuronError IO) ()
streamOfSizeFiltered f s =
  let batchSize = fromIntegral $ natVal (Proxy @b)
      substreams = S.chunksOf batchSize . S.filter f $ s
   in S.mapsM collapseSubstreams substreams
  where
    collapseSubstreams ::
      S.Stream
        (S.Of (V 10 Double, V 784 Double))
        (ExceptT HeuronError IO)
        x ->
      ExceptT HeuronError IO (S.Of (V b (V 10 Double), V b (V 784 Double)) x)
    collapseSubstreams substream = do
      labelsAndImages <- S.toList substream
      let collapsed = S.mapOf (S.bimap (mkV'' @b) (mkV'' @b) . unzip) labelsAndImages
      return collapsed
    mkV'' :: forall b a. (KnownNat b) => [a] -> V b a
    mkV'' as = case mkV @b as of
      Nothing -> error "mkV''"
      Just v -> v

streamOfSize ::
  forall b.
  (KnownNat b) =>
  S.Stream (S.Of (V 10 Double, V 784 Double)) (ExceptT HeuronError IO) () ->
  S.Stream (S.Of (Input b 10 Double, Input b 784 Double)) (ExceptT HeuronError IO) ()
streamOfSize = streamOfSizeFiltered @b (const True)
