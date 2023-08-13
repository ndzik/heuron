{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Main where

import Codec.Serialise
import Control.Lens
import Control.Monad.Except (ExceptT, runExceptT)
import qualified Data.ByteString.Lazy as BSL
import Data.Data (Proxy (..))
import Data.Functor ((<&>))
import Digits
import GHC.TypeLits (natVal)
import GHC.TypeNats (KnownNat)
import Heuron.Functions
import Heuron.V1
import Heuron.V1.Batched
import Linear.V
import Monomer
import Streaming (liftIO)
import qualified Streaming as S
import qualified Streaming.Prelude as S
import System.Random (getStdGen)
import Text.Printf (printf)
import Types
import View

pathToMNISTLabel :: FilePath
pathToMNISTLabel = "./data/train-labels-idx1-ubyte"

pathToMNISTImage :: FilePath
pathToMNISTImage = "./data/train-images-idx3-ubyte"

scaledBy :: (KnownNat n, KnownNat m) => Double -> Input n m Double -> Input n m Double
scaledBy x = fmap (fmap (* x))

main :: forall pixelCount batchSize numOfImages hiddenNeuronCount. (hiddenNeuronCount ~ 32, pixelCount ~ 784, batchSize ~ 100, numOfImages ~ 57321) => IO ()
main = do
  -- Describe network.
  rng <- getStdGen
  inputLayer <- Layer <$> (scaledBy (1 / 784) <$> randomM' @hiddenNeuronCount @pixelCount rng) <*> randomV' @hiddenNeuronCount rng <*> return zero
  hiddenLayer00 <- Layer <$> (scaledBy (1 / 32) <$> randomM' @hiddenNeuronCount @hiddenNeuronCount rng) <*> randomV' @hiddenNeuronCount rng <*> return zero
  hiddenLayer01 <- Layer <$> (scaledBy (1 / 32) <$> randomM' @hiddenNeuronCount @hiddenNeuronCount rng) <*> randomV' @hiddenNeuronCount rng <*> return zero
  outputLayer <- Layer <$> randomM' @10 @hiddenNeuronCount rng <*> randomV' @10 rng <*> return zero
  let learningRate = 0.25
      ann =
        inputLayer ReLU (StochasticGradientDescent learningRate)
          :>: hiddenLayer00 ReLU (StochasticGradientDescent learningRate)
          :>: hiddenLayer01 ReLU (StochasticGradientDescent learningRate)
            =| outputLayer Softmax (StochasticGradientDescent learningRate)
      initialTrainerState = TrainerState ann CategoricalCrossEntropy

  let producer env sendMsg = do
        -- Train network.
        labels <- streamMNISTLabels pathToMNISTLabel
        imgs <- streamMNISTImages @pixelCount pathToMNISTImage
        let onlyThree (label, _) = label ^?! ix 2 == 1
            s = streamOfSize @batchSize $ S.zip labels imgs
            trainNetwork (ts, epoch) (labels, images) = do
              (tr, ts') <- liftIO $ runTrainer (oneEpoch images labels) ts
              liftIO . sendMsg . HeuronUpdate $ UpdateEvent (tr ^. trainingResultLoss) (tr ^. trainingResultAccuracy) epoch images (ts' ^. network)
              return (ts', epoch + 1)

        print "Starting training..."
        _res <- runExceptT $ S.foldM_ trainNetwork (pure (initialTrainerState, 0)) pure s
        print "Done training."

  let config =
        [ appWindowTitle "Heuron",
          appTheme darkTheme,
          appScaleFactor 1.5,
          appInitEvent HeuronInit
        ]
      model =
        HeuronModel
          { _heuronModelNet = ann,
            _heuronModelAvgLoss = 0.00,
            _heuronModelAccuracy = 0.00,
            _heuronModelCurrentEpoch = 0,
            _heuronModelMaxEpochs = maxEpochs,
            _heuronModelCurrentBatch = zero
          }
  startApp model (handleEvent producer) buildUI config
  where
    maxEpochs = natVal (Proxy @numOfImages) `div` natVal (Proxy @batchSize)
    forwardNetwork (ts, epoch) (labels, images) = do
      (prediction, ts') <- liftIO $ runTrainer (trainForward images) ts
      liftIO . putStrLn $ printf "Network:\n%s" (show $ _network ts')
      liftIO . putStrLn $ printf "Prediction: %s" (show prediction)
      return (ts', epoch + 1)

handleEvent ::
  (WidgetEnv (HeuronModel b net) (HeuronEvent b net) -> ProducerHandler (HeuronEvent b net)) ->
  WidgetEnv (HeuronModel b net) (HeuronEvent b net) ->
  WidgetNode (HeuronModel b net) (HeuronEvent b net) ->
  HeuronModel b net ->
  HeuronEvent b net ->
  [AppEventResponse (HeuronModel b net) (HeuronEvent b net)]
handleEvent producer we node model evt = case evt of
  HeuronInit -> [Producer $ producer we]
  HeuronUpdate (UpdateEvent avgLoss acc currentEpoch cb newNet) ->
    [ Model
        model
          { _heuronModelAvgLoss = avgLoss,
            _heuronModelAccuracy = acc,
            _heuronModelCurrentEpoch = currentEpoch,
            _heuronModelCurrentBatch = cb,
            _heuronModelNet = newNet
          }
    ]

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
